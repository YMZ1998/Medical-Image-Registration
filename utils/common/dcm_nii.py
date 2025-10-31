import os

import SimpleITK as sitk

def load_and_save_dicom_as_nifti(dicom_dir, output_nii_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # 可选：检查像素值范围
    # import numpy as np
    # array = sitk.GetArrayFromImage(image)
    # print(f"Min = {np.min(array)}, Max = {np.max(array)}")

    sitk.WriteImage(image, output_nii_path)  # 保存为 .nii 或 .nii.gz

def load_all_dicom_series(input_dir, output_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(input_dir)
    print(series_ids)
    if not series_ids:
        print("❌ 没有找到任何 DICOM series。")
        return

    print(f"📦 找到 {len(series_ids)} 个 DICOM series.")

    # for i, series_id in enumerate(series_ids):
    #     series_files = reader.GetGDCMSeriesFileNames(input_dir, series_id)
    #     reader.SetFileNames(series_files)
    #     image = reader.Execute()
    #     image = sitk.Cast(image, sitk.sitkInt16)
    #     pixel_id = image.GetPixelIDTypeAsString()
    #     print(f"SimpleITK 数据类型: {pixel_id}")
    #     output_path = os.path.join(output_dir, f"series_{i}.nii.gz")
    #     sitk.WriteImage(image, output_path)
    #
    #     print(f"✅ 已保存: {output_path} (shape: {image.GetSize()})")

if __name__ == "__main__":
    input_dicom_dir = r'D:\Data\seg\open_atlas\test_atlas\LCTSC-Test-S2-201\IMAGES\dicom_series'
    output_dir = r'D:\Data\seg\open_atlas\test_atlas\LCTSC-Test-S2-201\IMAGES'
    os.makedirs(output_dir, exist_ok=True)
    load_all_dicom_series(input_dicom_dir, output_dir)

    # input_dicom_dir2 =r'D:\Data\MIR\images\云南省肿瘤\初始和EX\2a\7b\2a7b40452a134971c437ee54e88e6485e72755defb2bc8f07a8aa94a0621ab22\dicom'
    # input_dicom_dir2 =r'D:\Data\MIR\images\云南省肿瘤\初始和EX\15\d4\15d47f60b848333586c07fea0ac387880805ed5b3734c29db1090fcf2c048bac\dicom'
    input_dicom_dir2 =r'D:\Data\cbct\CT2\HEAD_ALL\HEAD\HEAD1'
    output_nii_path = r'D:\Data\cbct\CT2\HEAD_ALL\HEAD\HEAD12\CT.nii.gz'
    # load_and_save_dicom_as_nifti(input_dicom_dir2, output_nii_path)
