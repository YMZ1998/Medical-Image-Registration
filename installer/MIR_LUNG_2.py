import argparse
import os
import shutil
import warnings

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def normalize_image(array, min_v, max_v):
    array = np.clip(array, min_v, max_v)
    array = (array - min_v) / (max_v - min_v)
    return array


def load_image(image_path: str, spatial_size: tuple[int, int, int], normalize: bool = True):
    origin_image = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(origin_image).astype(np.float32)

    if normalize:
        array = normalize_image(array, -1200, 400)

    image = sitk.GetImageFromArray(array)
    image.CopyInformation(origin_image)
    image = resample_image(image, spatial_size)
    array = sitk.GetArrayFromImage(image)  # (D, H, W)

    return np.expand_dims(array, axis=0), image


def resample_image(image, target_size):
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


def save_ddf(array, file_path, origin_image, reference: sitk.Image):
    arr = np.asarray(array).astype(np.float32, copy=False)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")
    # 支持 (3, D, H, W) 或 (D, H, W, 3)
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported DDF array shape {arr.shape}. Expect (3,D,H,W) or (D,H,W,3).")

    sitk_image = sitk.GetImageFromArray(arr, isVector=True)
    sitk_image.SetSpacing(origin_image.GetSpacing())
    sitk_image.SetOrigin(origin_image.GetOrigin())
    sitk_image.SetDirection(origin_image.GetDirection())

    sitk_image.CopyInformation(origin_image)

    if list(sitk_image.GetSize()) != list(reference.GetSize()):
        print(f"  [DDF] Resampling from {sitk_image.GetSize()} → {reference.GetSize()}")
        # sitk_image = resample_vector_field(sitk_image, reference)

    sitk_image = sitk.Cast(sitk_image, sitk.sitkVectorFloat32)
    arr = sitk.GetArrayFromImage(sitk_image)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sitk.WriteImage(sitk_image, file_path)

    print(f"  [Saved] {file_path}\n"
          f"  size={sitk_image.GetSize()}, comps={sitk_image.GetNumberOfComponentsPerPixel()}, "
          f"type={sitk_image.GetPixelIDTypeAsString()}")


def save_array_as_nii(array, file_path, reference):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, reference.GetPixelIDValue())
    if reference is not None:
        sitk_image = resample_image(sitk_image, reference.GetSize())
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def val_onnx(args):
    warnings.filterwarnings("ignore")

    print("Loading and preprocessing images...")
    fixed, fixed_image = load_image(args.fixed_path, args.image_size)
    moving, moving_image = load_image(args.moving_path, args.image_size)
    original_moving, _ = load_image(args.moving_path, args.image_size, normalize=False)

    print("  [Fixed],", np.mean(fixed))
    print("  [Moving],", np.mean(moving))

    input_tensor = np.concatenate([moving, fixed, original_moving], axis=0)  # shape: (3, D, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # shape: (1, 3, D, H, W)

    print(f"Input tensor shape: {input_tensor.shape}")

    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])

    print("Running inference...")
    ort_inputs = {"input": input_tensor}
    moved_np, ddf_np = ort_session.run(None, ort_inputs)  # shapes: (1, 1, D, H, W), (1, 3, D, H, W)

    ddf_array = ddf_np[0]  # shape: (C, Z, Y, X)
    ref_image = sitk.ReadImage(args.fixed_path)
    save_ddf(ddf_array, os.path.join(args.result_path, 'ddf_field2.mhd'), fixed_image, reference=ref_image)

    print(f"Moved output shape: {moved_np.shape}, DDF shape: {ddf_np.shape}")
    print(f"  [DDF] max={np.max(ddf_np):.4f}, min={np.min(ddf_np):.4f}, abs_mean={np.mean(ddf_np):.4f}")

    # print("Visualizing results...")
    # from utils.visualization_numpy import visualize_one_case
    # visualize_one_case({"fixed_image": fixed, "moving_image": moving}, moved_np, ddf_np)

    print("Saving results...")
    # remove_and_create_dir(args.result_path)

    pred_array = moved_np[0, 0]
    ref_image = sitk.ReadImage(args.fixed_path)

    save_array_as_nii(pred_array, os.path.join(args.result_path, args.file_name), reference=ref_image)
    print("ONNX inference complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX model for registration purposes")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint/mir_lung2.onnx', help="Path to ONNX model file")
    parser.add_argument('--fixed_path', type=str, default='./data/fixed.nii.gz', help="Path to fixed (reference) image")
    parser.add_argument('--moving_path', type=str, default='./data/moving.nii.gz', help="Path to moving image")
    parser.add_argument('--result_path', type=str, default='./result', help="Directory to save the prediction result")
    parser.add_argument('--file_name', type=str, default='predict.nii.gz', help="Path to save results")
    parser.add_argument('--image_size', type=tuple, default=(192, 192, 192), help="Input image size as a tuple")

    args = parser.parse_args()
    print("Arguments:", args)

    import datetime
    import time

    start = time.time()
    val_onnx(args)
    print("Consume time:", str(datetime.timedelta(seconds=int(time.time() - start))))
