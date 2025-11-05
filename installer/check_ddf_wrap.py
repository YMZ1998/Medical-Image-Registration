import os

import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.blocks import Warp


def resample_image(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    original_size = np.array(image.GetSize())
    target_size = np.array(target_size)
    original_spacing = np.array(image.GetSpacing())
    new_spacing = original_spacing * (original_size / target_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputSpacing([float(sp) for sp in new_spacing])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    return resampler.Execute(image)


def sitk_to_torch(image: sitk.Image, device="cpu"):
    """Convert SimpleITK image to torch tensor (NCHWD format)."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)  # [D, H, W]
    tensor = torch.from_numpy(arr)[None, None, ...].to(device)  # -> [1, 1, D, H, W]
    return tensor


def torch_to_sitk(tensor: torch.Tensor, reference: sitk.Image):
    """Convert torch tensor (NCHWD) back to SimpleITK image and copy spatial info."""
    arr = tensor.squeeze().cpu().numpy().astype(np.float32)
    image = sitk.GetImageFromArray(arr)
    image.CopyInformation(reference)
    return image


def show_img(img, slice_idx, ax, axis=0, title=None):
    """
    显示单个切片 (origin='lower' 保证方向一致)
    axis: 0=axial, 1=coronal, 2=sagittal
    """
    img_np = img.cpu().numpy()
    if axis == 0:  # Axial (Z)
        slice_img = img_np[slice_idx, :, :]
    elif axis == 1:  # Coronal (Y)
        slice_img = img_np[:, slice_idx, :]
    else:  # Sagittal (X)
        slice_img = img_np[:, :, slice_idx]

    ax.imshow(slice_img, cmap="gray", origin="lower")
    if title:
        ax.set_title(title)
    ax.axis("off")


def visualize_one_case(check_data, pred_image, axis=0):
    # === 数据准备 ===
    def prep_image(img):
        if isinstance(img, torch.Tensor):
            return img[0, 0].cpu()
        else:
            import SimpleITK as sitk
            arr = np.array(sitk.GetArrayFromImage(img), dtype=np.float32)
            return torch.from_numpy(arr)

    fixed_image = prep_image(check_data["fixed_image"])
    moving_image = prep_image(check_data["moving_image"])
    pred_image = pred_image[0, 0].cpu()

    # === 选取切片索引 ===
    num_slices = fixed_image.shape[axis]
    slice_indices = range(num_slices // 4, num_slices * 3 // 4, 10)

    for slice_idx in slice_indices:
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))
        axs = axs.ravel()

        show_img(fixed_image, slice_idx, axs[0], axis, "Fixed")
        show_img(moving_image - fixed_image, slice_idx, axs[1], axis, "Moving-Diff")
        show_img(pred_image - fixed_image, slice_idx, axs[2], axis, "Warped-Diff")
        show_img(pred_image, slice_idx, axs[3], axis, "Warped")

        plt.suptitle(f"Slice {slice_idx} (axis={axis})", fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"
    ddf_path = "./result/ddf_field.mhd"
    output_path = "./result/moved_image.nii.gz"
    device = "cpu"  # or "cuda"

    print("Loading images...")
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # fixed_image = resample_image(fixed_image, (192, 192, 192))
    # moving_image = resample_image(moving_image, (192, 192, 192))

    ddf_field = sitk.ReadImage(ddf_path, sitk.sitkVectorFloat64)
    ddf_arr = sitk.GetArrayFromImage(ddf_field).astype(np.float32)
    print(f"DDF stats -> min: {ddf_arr.min():.4f}, max: {ddf_arr.max():.4f}, mean: {ddf_arr.mean():.4f}")

    moving_tensor = sitk_to_torch(moving_image, device)
    ddf_tensor = torch.from_numpy(ddf_arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    print(f"DDF tensor shape: {ddf_tensor.shape}")

    warp_layer = Warp(mode="bilinear", padding_mode="border").to(device)
    with torch.no_grad():
        moved_tensor = warp_layer(moving_tensor, ddf_tensor)

    check_data = {"fixed_image": fixed_image, "moving_image": moving_image}
    print("Visualizing warped result...")
    visualize_one_case(check_data, moved_tensor, 2)

    moved_image = torch_to_sitk(moved_tensor, fixed_image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(moved_image, output_path)

    print(f"✅ Moved image saved to: {output_path}")


if __name__ == "__main__":
    main()
