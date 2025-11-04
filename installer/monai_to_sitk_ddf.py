import SimpleITK as sitk
import numpy as np

from installer.check_ddf_wrap import resample_image


def monai_ddf_to_sitk(ddf_field, fixed_image):
    """
    ddf_np: numpy array, shape (D, H, W, 3) or (3, D, H, W)
            MONAI 输出的 voxel-level 位移（像素单位）
    fixed_image: sitk.Image, 用于参考 spacing, origin, direction
    """
    ddf_np = sitk.GetArrayFromImage(ddf_field)
    if ddf_np.shape[0] == 3:  # (C, D, H, W) -> (D, H, W, C)
        ddf_np = np.moveaxis(ddf_np, 0, -1)

    spacing = np.array(fixed_image.GetSpacing())
    # 转换为物理位移
    ddf_phys = ddf_np * spacing[np.newaxis, np.newaxis, np.newaxis, :]
    ddf_sitk = sitk.GetImageFromArray(ddf_phys, isVector=True)
    ddf_sitk.SetOrigin(fixed_image.GetOrigin())
    ddf_sitk.SetSpacing(fixed_image.GetSpacing())
    ddf_sitk.SetDirection(fixed_image.GetDirection())
    return ddf_sitk


if __name__ == "__main__":
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"
    ddf_path = "./result/ddf_field.mhd"

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    ddf_field = sitk.ReadImage(ddf_path)
    ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    fixed_image = resample_image(fixed_image, (192, 192, 192))
    moving_image = resample_image(moving_image, (192, 192, 192))
    ddf = monai_ddf_to_sitk(ddf_field, fixed_image)

    sitk.WriteImage(ddf, r'D:\code\Medical-Image-Registration\installer\result\ddf.nii.gz')
