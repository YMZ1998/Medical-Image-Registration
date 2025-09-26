import os
import sys
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon


def load_ct_series(ct_path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_path)
    print(series_ids)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {ct_path}")
    series_files = reader.GetGDCMSeriesFileNames(ct_path, series_ids[0])
    reader.SetFileNames(series_files)
    return reader.Execute()


def rtstruct_to_masks(rtstruct_path, ct_image, out_folder):
    ds = pydicom.dcmread(rtstruct_path)
    spacing = ct_image.GetSpacing()
    origin = ct_image.GetOrigin()
    direction = ct_image.GetDirection()
    size = ct_image.GetSize()
    array_shape = (size[2], size[1], size[0])  # z, y, x

    os.makedirs(out_folder, exist_ok=True)

    for roi, contour in zip(ds.StructureSetROISequence,
                            ds.ROIContourSequence):
        roi_name = roi.ROIName.strip().replace(" ", "_")
        print(f"Processing ROI: {roi_name}")
        mask = np.zeros(array_shape, dtype=np.uint8)

        for c in contour.ContourSequence:
            pts = np.array(c.ContourData).reshape(-1, 3)
            idx = [ct_image.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
            idx = np.array(idx)
            if len(np.unique(idx[:, 2])) != 1:
                continue
            z = int(np.round(np.mean(idx[:, 2])))
            rr, cc = polygon(idx[:, 1], idx[:, 0])
            mask[z, rr, cc] = 1

        mask_img = sitk.GetImageFromArray(mask)
        mask_img.SetSpacing(spacing)
        mask_img.SetOrigin(origin)
        mask_img.SetDirection(direction)

        out_path = os.path.join(out_folder, f"{roi_name}.nii.gz")
        sitk.WriteImage(mask_img, out_path)
        print(f"Saved: {out_path}")

def find_rtstruct_dcm(ct_dir):
    for root, dirs, files in os.walk(ct_dir):
        for file in files:
            if file.endswith(".dcm") and "RTSTRUCT" in file.upper():
                return os.path.join(root, file)

if __name__ == "__main__":
    ct_dir = r"C:\Users\Admin\Desktop\17-2320_JIANGTAO"
    rtstruct_dcm = find_rtstruct_dcm(ct_dir)
    print(rtstruct_dcm)
    out_dir = r"C:\Users\Admin\Desktop\\nii"

    print("Starting RTSTRUCT to NIfTI conversion...")
    ct_image = load_ct_series(ct_dir)
    sitk.WriteImage(ct_image, os.path.join(out_dir, "image.nii.gz"))
    rtstruct_to_masks(rtstruct_dcm, ct_image, out_dir)
    print("✅ Conversion finished.")
