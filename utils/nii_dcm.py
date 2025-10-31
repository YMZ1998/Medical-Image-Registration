import datetime
import os
import random
import shutil
import uuid

import SimpleITK as sitk
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian


def nii_to_dicom_series(nii_path, out_dir, use_random_id=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 读取 NIfTI
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)  # shape: [z, y, x]
    nx, ny, nz = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print("NIfTI shape (x,y,z):", nx, ny, nz)
    print("Spacing:", spacing)
    print("Origin:", origin)
    print("Direction:", direction)

    # -----------------------------
    # 统一 UID
    study_uid = generate_uid()
    series_uid = generate_uid()
    if use_random_id:
        patient_id = str(random.randint(10 ** 7, 10 ** 8 - 1))
        patient_name = f"{uuid.uuid4().hex[:8]}^{uuid.uuid4().hex[:8]}"
    else:
        patient_id = "12345678"
        patient_name = "TEST^4DCT"
    print("Patient ID:", patient_id)
    print("Patient Name:", patient_name)
    today = datetime.datetime.now().strftime("%Y%m%d")

    # -----------------------------
    # 遍历切片
    for k in range(array.shape[0]):
        slice_raw = array[k, :, :]  # z方向切片
        arr = np.rint(slice_raw).astype(np.int16)  # 转 int16

        # ImagePositionPatient
        z_pos = origin[2] + k * spacing[2]
        ipp = [origin[0], origin[1], z_pos]

        # ImageOrientationPatient
        iop = [direction[0], direction[1], 0, direction[3], direction[4], 0]

        # -----------------------------
        # pydicom 文件元数据
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        filename = os.path.join(out_dir, f"slice_{k + 1:03d}.dcm")
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # 基本信息
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = today

        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        # 图像尺寸
        rows, cols = arr.shape
        ds.Rows = int(rows)
        ds.Columns = int(cols)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1

        # Pixel spacing & slice thickness
        ds.PixelSpacing = [str(spacing[1]), str(spacing[0])]
        ds.SliceThickness = str(spacing[2])
        ds.ImagePositionPatient = [str(float(v)) for v in ipp]
        ds.ImageOrientationPatient = [str(float(v)) for v in iop]

        ds.InstanceNumber = k + 1
        dt = datetime.datetime.now()
        ds.StudyDate = dt.strftime("%Y%m%d")
        ds.StudyTime = dt.strftime("%H%M%S")

        # PixelData
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().newbyteorder()
        ds.PixelData = arr.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # 保存
        ds.save_as(filename, write_like_original=False)

        if k == 0:
            print("First slice saved:", filename)
            print("InstanceNumber:", ds.InstanceNumber)
            print("SeriesInstanceUID:", ds.SeriesInstanceUID)
            print("StudyInstanceUID:", ds.StudyInstanceUID)
            print("Rows,Columns:", ds.Rows, ds.Columns)
            print("PixelSpacing:", ds.PixelSpacing)
            print("SliceThickness:", ds.SliceThickness)
            print("ImagePositionPatient:", ds.ImagePositionPatient)
            print("ImageOrientationPatient:", ds.ImageOrientationPatient)

    print(f"Done, wrote {array.shape[0]} slices to {out_dir}")


def get_image_basename(path: str) -> str:
    """返回医学图像文件名（去掉路径和扩展名，如 fixed.nii.gz → fixed）"""
    filename = os.path.basename(path)
    for ext in [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]  # fallback


if __name__ == "__main__":
    nii_path = r"D:\code\Medical-Image-Registration\installer\data2\fixed.nii.gz"
    out_dir = os.path.join(os.path.dirname(nii_path), get_image_basename(nii_path))
    nii_to_dicom_series(nii_path, out_dir, 0)
