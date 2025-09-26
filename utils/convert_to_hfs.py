import SimpleITK as sitk


def convert_to_hfs(mask_path, out_path):
    mask_img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask_img)  # [z, y, x]

    # 获取原始信息
    spacing = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    direction = mask_img.GetDirection()  # tuple, len=9

    print("Original spacing:", spacing)
    print("Original direction:", direction)
    print("Original origin:", origin)

    # --- 假设原图是 LPS (常见 NIfTI) ---
    # HFS <-> LPS: X翻转，Y翻转
    arr_hfs = arr[:, ::-1, ::-1]  # Y轴、X轴翻转

    # 创建新图像
    mask_hfs = sitk.GetImageFromArray(arr_hfs)
    mask_hfs.SetSpacing(spacing)
    mask_hfs.SetOrigin(origin)

    # 设置 HFS 方向矩阵
    # HFS: X: L->R, Y: P->A, Z: I->S
    mask_hfs.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # 保存
    sitk.WriteImage(mask_hfs, out_path)
    print("Saved HFS mask to:", out_path)

if __name__ == "__main__":
    mask_path = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030\ct.nii.gz"
    out_path = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030\ct2.nii.gz"
    convert_to_hfs(mask_path, out_path)

