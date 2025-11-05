import argparse
import os
import shutil
import warnings
import datetime
import time

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort


def remove_and_create_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def normalize_image(array: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    array = np.clip(array, min_v, max_v)
    return (array - min_v) / (max_v - min_v)


def resample_image(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    original_size = np.array(image.GetSize())
    target_size = np.array(target_size)
    original_spacing = np.array(image.GetSpacing())
    new_spacing = original_spacing * (original_size / target_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputSpacing([float(sp) for sp in new_spacing])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    return resampler.Execute(image)


def load_image(image_path: str, spatial_size: tuple[int, int, int], normalize: bool = True):
    origin_image = sitk.ReadImage(image_path)
    # print(image_path)
    array = sitk.GetArrayFromImage(origin_image).astype(np.float32)

    if normalize:
        array = normalize_image(array, -1200, 400)

    image = sitk.GetImageFromArray(array)
    image.CopyInformation(origin_image)
    image = resample_image(image, spatial_size)
    # print(image.GetSize(), image.GetSpacing(), image.GetOrigin())
    array = sitk.GetArrayFromImage(image)  # (D, H, W)

    return np.expand_dims(array, axis=0), image  # shape: (1, D, H, W)


def resample_vector_field(ddf_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    original_size = np.array(ddf_image.GetSize())
    target_size = np.array(reference_image.GetSize())
    original_spacing = np.array(ddf_image.GetSpacing())
    new_spacing = original_spacing * (original_size / target_size)
    # print(original_size, target_size, original_spacing, new_spacing)
    ratio = new_spacing / original_spacing
    print("  ratio:", ratio)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputSpacing([float(sp) for sp in new_spacing])
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampled = resampler.Execute(ddf_image)

    # transfer monai to simpleitk
    arr = sitk.GetArrayFromImage(resampled)
    arr = arr[..., ::-1]
    spacing = np.array(resampled.GetSpacing())  # (sx, sy, sz)
    print(f" spacing: {spacing}")

    # 分别乘以 spacing 的三个分量
    arr[..., 0] = arr[..., 0] * spacing[0]
    arr[..., 1] = arr[..., 1] * spacing[1]
    arr[..., 2] = arr[..., 2] * spacing[2]
    resampled = sitk.GetImageFromArray(arr, isVector=True)
    resampled.CopyInformation(reference_image)
    return resampled


def save_ddf(array, file_path, origin_image, reference: sitk.Image):
    arr = np.asarray(array).astype(np.float32, copy=False)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")
    # 支持 (3, D, H, W) 或 (D, H, W, 3)
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
        print(f"  [DDF] shape={arr.shape}")
    elif arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported DDF array shape {arr.shape}. Expect (3,D,H,W) or (D,H,W,3).")

    # arr = arr[..., ::-1]
    # spacing = np.array(reference.GetSpacing())  # (sx, sy, sz)
    # print(f" spacing: {spacing}")
    #
    # # 分别乘以 spacing 的三个分量
    # ddf_phys = np.zeros_like(arr)
    # ddf_phys[..., 0] = arr[..., 0] * spacing[0]
    # ddf_phys[..., 1] = arr[..., 1] * spacing[1]
    # ddf_phys[..., 2] = arr[..., 2] * spacing[2]

    sitk_image = sitk.GetImageFromArray(arr, isVector=True)
    sitk_image.SetSpacing(origin_image.GetSpacing())
    sitk_image.SetOrigin(origin_image.GetOrigin())
    sitk_image.SetDirection(origin_image.GetDirection())

    sitk_image.CopyInformation(origin_image)

    if list(sitk_image.GetSize()) != list(reference.GetSize()):
        print(f"  [DDF] Resampling from {sitk_image.GetSize()} → {reference.GetSize()}")
        sitk_image = resample_vector_field(sitk_image, reference)

    # sitk_image.CopyInformation(reference)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkVectorFloat32)
    arr = sitk.GetArrayFromImage(sitk_image)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sitk.WriteImage(sitk_image, file_path)

    print(f"  [Saved] {file_path}\n"
          f"  size={sitk_image.GetSize()}, comps={sitk_image.GetNumberOfComponentsPerPixel()}, "
          f"type={sitk_image.GetPixelIDTypeAsString()}")


def val_onnx(args):
    warnings.filterwarnings("ignore")

    print("[Step 1] 加载与预处理图像...")
    fixed, fixed_image = load_image(args.fixed_path, args.image_size)
    moving, moving_image = load_image(args.moving_path, args.image_size)

    print("  [Fixed],", np.mean(fixed))
    print("  [Moving],", np.mean(moving))

    input_tensor = np.concatenate([moving, fixed], axis=0)  # (2, D, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # (1, 2, D, H, W)
    print(f"  Input tensor shape: {input_tensor.shape}")

    print("[Step 2] 加载 ONNX 模型...")
    ort_session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])

    print("[Step 3] 运行推理...")
    ort_inputs = {"input": input_tensor}
    [ddf_np] = ort_session.run(None, ort_inputs)
    print(f"  ddf_np shape: {ddf_np.shape}")

    print("[Step 4] 保存结果...")
    pred_array = ddf_np[0]  # shape: (C, Z, Y, X)
    ref_image = sitk.ReadImage(args.fixed_path)

    start = time.time()
    save_ddf(pred_array, os.path.join(args.result_path, args.file_name), fixed_image, reference=ref_image)
    print(f"  DDF 保存完成 (耗时: {datetime.timedelta(seconds=int(time.time() - start))})")
    print("  ONNX 推理完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX model for medical image registration")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint/mir_lung.onnx', help="Path to ONNX model file")
    parser.add_argument('--fixed_path', type=str, default='./data/fixed.nii.gz', help="Path to fixed (reference) image")
    parser.add_argument('--moving_path', type=str, default='./data/moving.nii.gz', help="Path to moving image")
    parser.add_argument('--result_path', type=str, default='./result', help="Directory to save prediction result")
    parser.add_argument('--file_name', type=str, default='ddf_field.mhd', help="Output DDF file name")
    parser.add_argument('--image_size', type=tuple, default=(192, 192, 192), help="Input image size (D, H, W)")

    args = parser.parse_args()
    print("Arguments:", args)

    start = time.time()
    val_onnx(args)
    print("\nTotal Time:", str(datetime.timedelta(seconds=int(time.time() - start))))
