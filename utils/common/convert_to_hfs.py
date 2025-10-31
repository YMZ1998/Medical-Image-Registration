import SimpleITK as sitk


def convert_to_hfs(path, out_path):
    image = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(image)  # [z, y, x]

    # 获取原始信息
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()  # tuple, len=9

    print("Original spacing:", spacing)
    print("Original direction:", direction)
    print("Original origin:", origin)

    # --- 假设原图是 LPS (常见 NIfTI) ---
    # HFS <-> LPS: X翻转，Y翻转
    arr_hfs = arr[:, ::-1, ::-1]  # Y轴、X轴翻转

    # 创建新图像
    image_hfs = sitk.GetImageFromArray(arr_hfs)
    image_hfs.SetSpacing(spacing)
    image_hfs.SetOrigin(origin)

    # 设置 HFS 方向矩阵
    # HFS: X: L->R, Y: P->A, Z: I->S
    image_hfs.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # 保存
    sitk.WriteImage(image_hfs, out_path)
    print("Saved HFS mask to:", out_path)

if __name__ == "__main__":
    path = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030\ct.nii.gz"
    out_path = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030\ct2.nii.gz"
    convert_to_hfs(path, out_path)

