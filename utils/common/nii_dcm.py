import datetime
import os
import random
import shutil
import uuid

import SimpleITK as sitk
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian


def get_image_basename(path: str) -> str:
    filename = os.path.basename(path)
    for ext in [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]


def nii_to_dicom_series(nii_path, out_dir, use_random_id=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Load NIfTI
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)  # [z, y, x]
    nx, ny, nz = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print("NIfTI shape:", array.shape)
    print("Spacing:", spacing)
    print("Origin:", origin)
    print("Direction:", direction)

    # -----------------------------
    # UID generation
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()  # MUST have for RT environment

    if use_random_id:
        patient_id = str(random.randint(10 ** 7, 10 ** 8 - 1))
        patient_name = f"{uuid.uuid4().hex[:8]}^{uuid.uuid4().hex[:8]}"
    else:
        patient_id = "12345678"
        patient_name = "TEST^NIITOCT"

    today = datetime.datetime.now().strftime("%Y%m%d")

    print("Patient:", patient_name, patient_id)

    # -----------------------------
    # Write slices
    num_slices = array.shape[0]

    for k in range(num_slices):
        slice_raw = array[k, :, :]
        arr = np.rint(slice_raw).astype(np.int16)

        # Patient geometry
        z_pos = origin[2] + k * spacing[2]
        ipp = [origin[0], origin[1], float(z_pos)]
        # Simplify orientation: axial
        iop = [direction[0], direction[1], direction[2],
               direction[3], direction[4], direction[5]]

        # File meta
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        filename = os.path.join(out_dir, f"CT_{k + 1:03d}.dcm")
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Basic
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = today

        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid

        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        dt = datetime.datetime.now()
        ds.StudyDate = today
        ds.StudyTime = dt.strftime("%H%M%S")
        ds.AcquisitionDate = ds.StudyDate
        ds.AcquisitionTime = ds.StudyTime

        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1

        # Image properties
        rows, cols = arr.shape
        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1

        # Geometry
        ds.PixelSpacing = [str(spacing[1]), str(spacing[0])]
        ds.SliceThickness = str(spacing[2])
        ds.ImagePositionPatient = [str(v) for v in ipp]
        ds.ImageOrientationPatient = [str(v) for v in iop]

        # CT HU mapping
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"

        # Recommended
        ds.BodyPartExamined = "CHEST"
        ds.PositionReferenceIndicator = ""

        # Pixel data
        ds.PixelData = arr.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.save_as(filename, write_like_original=False)

        if k == 0:
            print("First slice:", filename)
            print("FrameOfReferenceUID:", ds.FrameOfReferenceUID)

    print(f"✔ Done! {num_slices} slices saved to: {out_dir}")


if __name__ == "__main__":
    nii_path = r"D:\debug\LUNG1-226\IMAGES\CT.nii.gz"
    out_dir = os.path.join(os.path.dirname(nii_path), get_image_basename(nii_path))
    nii_to_dicom_series(nii_path, out_dir, use_random_id=True)
