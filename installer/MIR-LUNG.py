import argparse
import os
import shutil
import warnings
import datetime
import time

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


def load_image(image_path, spatial_size, normalize=True):
    image = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    if normalize:
        array = normalize_image(array, -1200, 400)

    image = sitk.GetImageFromArray(array)
    image = resample_image(image, spatial_size)
    array = sitk.GetArrayFromImage(image)  # (D, H, W)

    return np.expand_dims(array, axis=0), image  # shape: (1, D, H, W)


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


def resample_vector_field(
    ddf_image,
    reference_image
):
    try:
        # 1. 创建 ResampleImageFilter 实例
        resample = sitk.ResampleImageFilter()

        # 2. 设置参考图像。这将定义输出的尺寸、间距、原点和方向。
        # resample.SetReferenceImage(reference_image)
        resample.SetSize(reference_image.GetSize())
        resample.SetOutputSpacing(reference_image.GetSpacing())
        resample.SetOutputOrigin(reference_image.GetOrigin())
        # resample.SetOutputDirection(reference_image.GetDirection())

        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0.0)

        # 5. 执行重采样
        resampled_ddf = resample.Execute(ddf_image)

        return resampled_ddf

    except Exception as e:
        print(f"重采样向量场时发生错误: {e}")
        raise e


def save_ddf(array, file_path,origin_spacing, reference=None):
    # Normalize shapes: allow (3,D,H,W) or (D,H,W,3)
    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[0] == 3:
        # (3, D, H, W) -> (D, H, W, 3)
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported DDF array shape {arr.shape}. Expect (3,D,H,W) or (D,H,W,3).")

    arr = arr.astype(np.float32, copy=False)
    print(np.max(arr), np.min(arr))
    print("DDF abs mean:", np.mean(np.abs(arr)))
    # SimpleITK expects (z,y,x,vector) -> GetImageFromArray with isVector=True
    sitk_image = sitk.GetImageFromArray(arr, isVector=True)
    # print("DDF GetSpacing:", sitk_image.GetSpacing())
    # print("DDF GetOrigin:", sitk_image.GetOrigin())
    # print("DDF GetSize:", sitk_image.GetSize())
    # print("DDF GetDirection:", sitk_image.GetDirection())
    # print("reference GetDirection:", reference.GetDirection())

    sitk_image.SetSpacing(origin_spacing)
    sitk_image.SetOrigin(reference.GetOrigin())

    if reference is not None:
        if list(sitk_image.GetSize()) != list(reference.GetSize()):
            print(f"Resampling DDF from {sitk_image.GetSize()} to {reference.GetSize()}")
            sitk_image = resample_vector_field(sitk_image, reference)
        sitk_image.CopyInformation(reference)
    arr = sitk.GetArrayFromImage(sitk_image)
    print(np.max(arr), np.min(arr))
    print("DDF abs mean:", np.mean(np.abs(arr)))
    # print("DDF GetSpacing:", sitk_image.GetSpacing())
    # print("DDF GetOrigin:", sitk_image.GetOrigin())
    # print("DDF GetDirection:", sitk_image.GetDirection())
    sitk_image = sitk.Cast(sitk_image, sitk.sitkVectorFloat32)

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sitk.WriteImage(sitk_image, file_path)
    print(f"Saved DDF to {file_path}; size={sitk_image.GetSize()}, "
          f"comps={sitk_image.GetNumberOfComponentsPerPixel()}, "
          f"type={sitk_image.GetPixelIDTypeAsString()}")


def val_onnx(args):
    warnings.filterwarnings("ignore")

    print("Loading and preprocessing images...")
    fixed, fixed_image = load_image(args.fixed_path, args.image_size)
    moving, moving_image = load_image(args.moving_path, args.image_size)

    input_tensor = np.concatenate([moving, fixed], axis=0)  # shape: (2, D, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # shape: (1, 3, D, H, W)

    print(f"Input tensor shape: {input_tensor.shape}")

    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])

    print("Running inference...")
    ort_inputs = {"input": input_tensor}
    [ddf_np] = ort_session.run(None, ort_inputs)  # shapes: (1, 3, D, H, W)

    print(f"DDF shape: {ddf_np.shape}")
    print(np.max(ddf_np), np.min(ddf_np))
    print("Saving results...")
    # remove_and_create_dir(args.result_path)

    pred_array = ddf_np[0]
    print("Saving DDF...")
    print(pred_array.shape)
    ref_image = sitk.ReadImage(args.fixed_path)

    start = time.time()
    save_ddf(pred_array, os.path.join(args.result_path, args.file_name),fixed_image.GetSpacing(), reference=ref_image)
    print("Save DDF Consume time:", str(datetime.timedelta(seconds=int(time.time() - start))))
    print("ONNX inference complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX model for registration purposes")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint/mir_lung.onnx', help="Path to ONNX model file")
    parser.add_argument('--fixed_path', type=str, default='./data/fixed.nii.gz', help="Path to fixed (reference) image")
    parser.add_argument('--moving_path', type=str, default='./data/moving.nii.gz', help="Path to moving image")
    parser.add_argument('--result_path', type=str, default='./result', help="Directory to save the prediction result")
    parser.add_argument('--file_name', type=str, default='ddf_field.nii.gz', help="Path to save results")
    parser.add_argument('--image_size', type=tuple, default=(192, 192, 192), help="Input image size as a tuple")

    args = parser.parse_args()
    print("Arguments:", args)

    start = time.time()
    val_onnx(args)
    print("Consume time:", str(datetime.timedelta(seconds=int(time.time() - start))))
