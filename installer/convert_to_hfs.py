import SimpleITK as sitk


def convert_to_hfs(path):
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
    sitk.WriteImage(image_hfs, path)
    print("Saved HFS mask to:", path)

if __name__ == "__main__":
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"
    convert_to_hfs(fixed_image_path)
    convert_to_hfs(moving_image_path)

