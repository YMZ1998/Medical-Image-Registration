import os

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt

from installer.check_ddf_wrap import resample_image


def warp_image_with_ddf(fixed, moving, ddf_field):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)

    transform = sitk.DisplacementFieldTransform(ddf_field)
    resampler.SetTransform(transform)

    moved = resampler.Execute(moving)
    return moved


def show_images(fixed, moving, moved, arr_ddf, axis=0, slice_indices=None, step=20):
    arr_fixed = sitk.GetArrayFromImage(fixed)
    arr_moving = sitk.GetArrayFromImage(moving)
    arr_moved = sitk.GetArrayFromImage(moved)

    Z, H, W = arr_fixed.shape
    if slice_indices is None:
        slice_indices = [Z // 4, Z // 2, 3 * Z // 4] if axis == 0 else [H // 4, H // 2, 3 * H // 4] if axis == 1 else [
            W // 4, W // 2, 3 * W // 4]

    n_slices = len(slice_indices)
    n_cols = 6 if ddf_field is not None else 5
    fig, axs = plt.subplots(n_slices, n_cols, figsize=(6 * n_cols, 3 * n_slices))

    if n_slices == 1:
        axs = axs.reshape(1, n_cols)

    for i, slice_idx in enumerate(slice_indices):
        # 提取切片
        if axis == 0:  # Axial
            fixed_slice = arr_fixed[slice_idx, :, :]
            moving_slice = arr_moving[slice_idx, :, :]
            moved_slice = arr_moved[slice_idx, :, :]
            if ddf_field is not None:
                vx = arr_ddf[slice_idx, :, :, 0]
                vy = arr_ddf[slice_idx, :, :, 1]
        elif axis == 1:  # Coronal
            fixed_slice = arr_fixed[:, slice_idx, :]
            moving_slice = arr_moving[:, slice_idx, :]
            moved_slice = arr_moved[:, slice_idx, :]
            if ddf_field is not None:
                vx = arr_ddf[:, slice_idx, :, 0]
                vy = arr_ddf[:, slice_idx, :, 2]
        else:  # Sagittal
            fixed_slice = arr_fixed[:, :, slice_idx]
            moving_slice = arr_moving[:, :, slice_idx]
            moved_slice = arr_moved[:, :, slice_idx]
            if ddf_field is not None:
                vx = arr_ddf[:, :, slice_idx, 1]
                vy = arr_ddf[:, :, slice_idx, 2]

        axs[i, 0].imshow(fixed_slice, cmap="gray", origin="lower")
        axs[i, 0].set_title(f"Fixed {slice_idx}")

        axs[i, 1].imshow(moving_slice, cmap="gray", origin="lower")
        axs[i, 1].set_title(f"Moving  {slice_idx}")

        axs[i, 2].imshow(moving_slice - fixed_slice, cmap="gray", origin="lower")
        axs[i, 2].set_title(f"Moving - Fixed")

        axs[i, 3].imshow(moved_slice - fixed_slice, cmap="gray", origin="lower")
        axs[i, 3].set_title(f"Moved - Fixed")

        axs[i, 4].imshow(moved_slice - moving_slice, cmap="gray", origin="lower")
        axs[i, 4].set_title(f"Moved - Moving")

        # ====== 新增部分：显示 DDF 向量场 ======
        if ddf_field is not None:
            axs[i, 5].imshow(moved_slice, cmap="gray", origin="lower")
            axs[i, 5].set_title(f"Displacement Field")

            # 采样稀疏点
            y, x = np.mgrid[0:vx.shape[0]:step, 0:vx.shape[1]:step]
            u = vx[::step, ::step]
            v = vy[::step, ::step]
            axs[i, 5].quiver(x, y, v, u, color="r", angles="xy", scale_units="xy", scale=1)

        for j in range(n_cols):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"
    ddf_path = "./result/ddf_field.mhd"
    # ddf_path = "./result/ddf.nii.gz"
    # moving_image_path = r"D:\debug\moving_vol.nii.gz"
    # fixed_image_path = r"D:\debug\fixed_vol.nii.gz"
    # ddf_path = r"D:\debug\deformation_field.nii.gz"

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    ddf_field = sitk.ReadImage(ddf_path)
    ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    ddf_np = sitk.GetArrayFromImage(ddf_field)
    print(f"ddf_np shape: {ddf_np.shape}")
    if ddf_np.shape[0] == 3:  # (C, D, H, W) -> (D, H, W, C)
        ddf_np = np.moveaxis(ddf_np, 0, -1)
    # ddf_np = ddf_np[..., [0, 1, 2]]
    # fixed_image = resample_image(fixed_image, (192, 192, 192))
    # moving_image = resample_image(moving_image, (192, 192, 192))
    # new_spacing = [1, 1, 1]
    # fixed_image.SetSpacing(new_spacing)
    # moving_image.SetSpacing(new_spacing)
    # ddf_field.SetSpacing(new_spacing)

    print("fixed_image shape", fixed_image.GetSize(), fixed_image.GetSpacing(), fixed_image.GetDirection())
    print("moving_image shape", moving_image.GetSize(), moving_image.GetSpacing(), moving_image.GetDirection())
    print("ddf_field shape", ddf_field.GetSize(), ddf_field.GetSpacing(), ddf_field.GetDirection())

    moved_image = warp_image_with_ddf(fixed_image, moving_image, ddf_field)

    output_path = "./result/moved_image.nii.gz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(moved_image, output_path)
    print(f"✅ Moved image saved to: {output_path}")

    # 可视化多切片
    show_images(fixed_image, moving_image, moved_image, ddf_np, axis=0,
                slice_indices=[50, 60, 70, 80, 100])
    show_images(fixed_image, moving_image, moved_image, ddf_np, axis=1,
                slice_indices=[50, 60, 70, 80, 100])
    show_images(fixed_image, moving_image, moved_image, ddf_np, axis=2,
                slice_indices=[50, 60, 70, 80, 100])
