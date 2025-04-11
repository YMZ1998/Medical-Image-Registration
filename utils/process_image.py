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


def save_array_as_nii(array, file_path, reference, pixel_type=sitk.sitkInt16):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, pixel_type)
    if reference is not None:
        sitk_image = resample_image(sitk_image, reference.GetSize())
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)
