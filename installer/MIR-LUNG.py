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


def load_image(image_path, spatial_size, normalize=True):
    image = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    if normalize:
        array = normalize_image(array, -1200, 400)

    image = sitk.GetImageFromArray(array)
    image = resample_image(image, spatial_size)
    array = sitk.GetArrayFromImage(image)  # (D, H, W)

    return np.expand_dims(array, axis=0)  # shape: (1, D, H, W)


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
    fixed = load_image(args.fixed_path, args.image_size)
    moving = load_image(args.moving_path, args.image_size)
    original_moving = load_image(args.moving_path, args.image_size, normalize=False)

    input_tensor = np.concatenate([moving, fixed, original_moving], axis=0)  # shape: (3, D, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # shape: (1, 3, D, H, W)

    print(f"Input tensor shape: {input_tensor.shape}")

    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])

    print("Running inference...")
    ort_inputs = {"input": input_tensor}
    moved_np, ddf_np = ort_session.run(None, ort_inputs)  # shapes: (1, 1, D, H, W), (1, 3, D, H, W)

    print(f"Moved output shape: {moved_np.shape}, DDF shape: {ddf_np.shape}")

    # print("Visualizing results...")
    # from utils.visualization_numpy import visualize_one_case
    # visualize_one_case({"fixed_image": fixed, "moving_image": moving}, moved_np, ddf_np)

    print("Saving results...")
    remove_and_create_dir(args.result_path)

    pred_array = moved_np[0, 0]
    ref_image = sitk.ReadImage(args.fixed_path)

    save_array_as_nii(pred_array, os.path.join(args.result_path, args.file_name), reference=ref_image)
    print("ONNX inference complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX model for registration purposes")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint/mir_lung.onnx', help="Path to ONNX model file")
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
