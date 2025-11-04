import SimpleITK as sitk
from matplotlib import pyplot as plt

from installer.check_ddf_wrap import resample_image


def warp_image_with_ddf(fixed, moving, ddf_field):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)

    transform = sitk.DisplacementFieldTransform(ddf_field)
    resampler.SetTransform(transform)

    # ddf = sitk.TransformToDisplacementField(
    #     transform,
    #     size=fixed.GetSize(),
    #     outputSpacing=fixed.GetSpacing(),
    #     outputOrigin=fixed.GetOrigin(),
    #     outputDirection=fixed.GetDirection()
    # )
    # sitk.WriteImage(ddf, r'D:\code\Medical-Image-Registration\installer\result\ddf.nii.gz')

    moved = resampler.Execute(moving)
    return moved


def show_images(fixed, moving, moved, axis=0, slice_indices=None):
    arr_fixed = sitk.GetArrayFromImage(fixed)
    arr_moving = sitk.GetArrayFromImage(moving)
    arr_moved = sitk.GetArrayFromImage(moved)

    Z, H, W = arr_fixed.shape
    if slice_indices is None:
        slice_indices = [Z // 4, Z // 2, 3 * Z // 4] if axis == 0 else [H // 4, H // 2, 3 * H // 4] if axis == 1 else [
            W // 4, W // 2, 3 * W // 4]

    n_slices = len(slice_indices)
    fig, axs = plt.subplots(n_slices, 3, figsize=(9, 3 * n_slices))

    if n_slices == 1:
        axs = axs.reshape(1, 3)

    for i, slice_idx in enumerate(slice_indices):
        # 提取切片
        if axis == 0:  # Axial
            fixed_slice = arr_fixed[slice_idx, :, :]
            moving_slice = arr_moving[slice_idx, :, :]
            moved_slice = arr_moved[slice_idx, :, :]
        elif axis == 1:  # Coronal
            fixed_slice = arr_fixed[:, slice_idx, :]
            moving_slice = arr_moving[:, slice_idx, :]
            moved_slice = arr_moved[:, slice_idx, :]
        else:  # Sagittal
            fixed_slice = arr_fixed[:, :, slice_idx]
            moving_slice = arr_moving[:, :, slice_idx]
            moved_slice = arr_moved[:, :, slice_idx]

        axs[i, 0].imshow(fixed_slice, cmap="gray", origin="lower")
        axs[i, 0].set_title(f"Fixed Image Slice {slice_idx}")
        axs[i, 1].imshow(moving_slice - fixed_slice, cmap="gray", origin="lower")
        axs[i, 1].set_title(f"Moving Image Slice {slice_idx}")
        axs[i, 2].imshow(moved_slice - fixed_slice, cmap="gray", origin="lower")
        axs[i, 2].set_title(f"Moved Image Slice {slice_idx}")

        for ax in axs[i, :]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"
    # ddf_path = "./result/ddf_field.mhd"
    ddf_path = "./result/ddf.nii.gz"

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    ddf_field = sitk.ReadImage(ddf_path)
    ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    fixed_image = resample_image(fixed_image, (192, 192, 192))
    moving_image = resample_image(moving_image, (192, 192, 192))

    print("fixed_image shape", fixed_image.GetSize(), fixed_image.GetSpacing(), fixed_image.GetDirection())
    print("moving_image shape", moving_image.GetSize(), moving_image.GetSpacing(), moving_image.GetDirection())
    print("ddf_field shape", ddf_field.GetSize(), ddf_field.GetSpacing(), ddf_field.GetDirection())

    moved_image = warp_image_with_ddf(fixed_image, moving_image, ddf_field)

    # output_path = "./result/moved_image.nii.gz"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # sitk.WriteImage(moved_image, output_path)
    # print(f"✅ Moved image saved to: {output_path}")

    # 可视化多切片
    show_images(fixed_image, moving_image, moved_image, axis=2, slice_indices=[50, 60, 70, 80, 90, 100])
