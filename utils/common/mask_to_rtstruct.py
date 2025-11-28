import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
from skimage import measure

def masks_to_rtstruct(dicom_dir, mask_files, labels, output_rtstruct, smooth_sigma=1.0):
    """
    将多个 mask 生成 RTSTRUCT，每个 mask 每层轮廓自动生成 polygon
    dicom_dir: 原始 CT DICOM 文件夹
    mask_files: mask 文件列表
    labels: mask 对应结构名称
    output_rtstruct: 输出 RTSTRUCT 路径
    smooth_sigma: 平滑 mask 的高斯 sigma
    """

    # 1️⃣ 读取 DICOM 序列
    dicom_files = sorted([os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)])
    if len(dicom_files) == 0:
        raise ValueError("DICOM 文件夹为空！")
    ref_ds = pydicom.dcmread(dicom_files[0])

    # 用 SimpleITK 读取整个 CT 序列
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(dicom_names)
    ct_volume = reader.Execute()
    spacing = ct_volume.GetSpacing()  # x, y, z
    origin = ct_volume.GetOrigin()
    direction = np.array(ct_volume.GetDirection()).reshape(3, 3)  # 3x3 matrix

    # 2️⃣ 创建 FileDataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    rtstruct = FileDataset(output_rtstruct, {}, file_meta=file_meta,
                           preamble=b"\0"*128, is_little_endian=True, is_implicit_VR=True)

    # 3️⃣ 设置基础信息
    rtstruct.PatientName = ref_ds.PatientName
    rtstruct.PatientID = ref_ds.PatientID
    rtstruct.StudyInstanceUID = ref_ds.StudyInstanceUID
    rtstruct.SeriesInstanceUID = pydicom.uid.generate_uid()
    rtstruct.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    rtstruct.SOPClassUID = file_meta.MediaStorageSOPClassUID
    rtstruct.StructureSetLabel = "TotalSegmentator"
    rtstruct.StructureSetDate = datetime.datetime.now().strftime("%Y%m%d")
    rtstruct.StructureSetTime = datetime.datetime.now().strftime("%H%M%S")
    rtstruct.Modality = "RTSTRUCT"

    # 4️⃣ StructureSetROISequence
    rtstruct.StructureSetROISequence = []
    for i, label in enumerate(labels):
        roi = Dataset()
        roi.ROINumber = i + 1
        roi.ROIName = label
        roi.ROIDescription = label
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        rtstruct.StructureSetROISequence.append(roi)

    # 5️⃣ ROIContourSequence
    rtstruct.ROIContourSequence = []
    for i, mask_file in enumerate(mask_files):
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = i + 1
        roi_contour.ContourSequence = []

        # 读取 mask
        mask = sitk.ReadImage(mask_file)
        if smooth_sigma > 0:
            mask = sitk.SmoothingRecursiveGaussian(mask, smooth_sigma)
        mask = sitk.BinaryThreshold(mask, 0.5, 1.0, 1, 0)
        arr = sitk.GetArrayFromImage(mask)  # z,y,x

        # 遍历每一层
        for z in range(arr.shape[0]):
            slice_mask = arr[z, :, :]
            contours = measure.find_contours(slice_mask, 0.5)

            for contour in contours:
                # 将 (y,x) -> 患者坐标 (mm)
                patient_coords = []
                for y, x in contour:
                    coord = origin + direction.dot([x*spacing[0], y*spacing[1], z*spacing[2]])
                    patient_coords.extend(coord.tolist())

                # 创建 ContourSequence item
                contour_item = Dataset()
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = len(contour)
                contour_item.ContourData = patient_coords
                roi_contour.ContourSequence.append(contour_item)

        rtstruct.ROIContourSequence.append(roi_contour)

    # 6️⃣ 保存 RTSTRUCT
    pydicom.dcmwrite(output_rtstruct, rtstruct)
    print("平滑 RTSTRUCT 已生成:", output_rtstruct)


if __name__ == "__main__":
    dicom_dir = r"C:\Users\Admin\Desktop\17-2320_JIANGTAO"
    src = r"D:\debug\segmentations"

    # 获取前10个 mask 文件
    mask_files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith(".nii.gz")][:10]
    labels = [os.path.splitext(os.path.basename(f))[0] for f in mask_files]

    output_rtstruct = os.path.join(dicom_dir, "rtstruct_demo.dcm")

    masks_to_rtstruct(dicom_dir, mask_files, labels, output_rtstruct)
