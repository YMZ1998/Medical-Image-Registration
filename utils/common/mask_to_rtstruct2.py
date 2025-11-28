import os

import SimpleITK as sitk
import numpy as np
from rt_utils import RTStructBuilder


def get_image_basename(path: str) -> str:
    filename = os.path.basename(path)
    for ext in [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]  # fallback


dicom_dir =  r"D:\debug\LUNG1-226\IMAGES\CT"
mask_folder = r"D:\debug\LUNG1-226\STRUCTURES"
output_rtstruct = os.path.join(dicom_dir, "rtstruct.dcm")

reader = sitk.ImageSeriesReader()
series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
print("DICOM 切片数:", len(dicom_names))

mask_files = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(".nii.gz")][:20]
labels = [get_image_basename(f) for f in mask_files]

rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_dir)

for mask_file, label in zip(mask_files, labels):
    img = sitk.ReadImage(mask_file)

    mask_from_sitkImage_zyx = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    mask_from_sitkImage_xzy = np.transpose(mask_from_sitkImage_zyx, axes=(2, 0, 1))
    mask_from_sitkImage_xyz = np.transpose(mask_from_sitkImage_xzy, (2, 1, 0))
    mask_from_sitkImage_int64 = mask_from_sitkImage_xyz
    mask_bool = mask_from_sitkImage_int64.astype(bool)

    rtstruct.add_roi(
        mask=mask_bool,
        name=label,
        approximate_contours=False,
        use_pin_hole=True
    )

rtstruct.save(output_rtstruct)
print("Saved RTSTRUCT:", output_rtstruct)
