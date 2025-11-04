import os

import SimpleITK as sitk
import numpy as np


def resample_image(image: sitk.Image, target_size: tuple) -> sitk.Image:
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


def save_ddf(array, file_path, origin_image, reference: sitk.Image):
    arr = np.asarray(array)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
        arr = arr.transpose(2, 1, 0, 3)
        print(f"  [DDF] shape={arr.shape}")
    elif arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported DDF array shape {arr.shape}. Expect (3,D,H,W) or (D,H,W,3).")

    arr = arr.astype(np.float32, copy=False)

    sitk_image = sitk.GetImageFromArray(arr, isVector=True)
    sitk_image.SetSpacing(origin_image.GetSpacing())
    sitk_image.SetOrigin(origin_image.GetOrigin())
    sitk_image.SetDirection(origin_image.GetDirection())

    sitk_image.CopyInformation(origin_image)

    if list(sitk_image.GetSize()) != list(reference.GetSize()):
        print(f"  [DDF] Resampling from {sitk_image.GetSize()} → {reference.GetSize()}")
        # sitk_image = resample_vector_field(sitk_image, reference)

    # sitk_image.CopyInformation(reference)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkVectorFloat32)
    arr = sitk.GetArrayFromImage(sitk_image)
    print(f"  [DDF] max={np.max(arr):.4f}, min={np.min(arr):.4f}, abs_mean={np.mean(arr):.4f}")

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sitk.WriteImage(sitk_image, file_path)

    print(f"  [Saved] {file_path}\n"
          f"  size={sitk_image.GetSize()}, comps={sitk_image.GetNumberOfComponentsPerPixel()}, "
          f"type={sitk_image.GetPixelIDTypeAsString()}")


def normalize_image(array: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    array = np.clip(array, min_v, max_v)
    return (array - min_v) / (max_v - min_v)


def load_image_sitk(image_path: str, spatial_size: tuple[int, int, int], normalize: bool = True):
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
