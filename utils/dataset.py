import json
import os


def overlay_img(img1, img2, slice_idx, ax, title=None):
    ax.imshow(1 - img1[:, slice_idx, :].T, cmap="Blues", origin="lower")
    ax.imshow(1 - img2[:, slice_idx, :].T, cmap="Oranges", origin="lower", alpha=0.5)
    if title is not None:
        ax.title.set_text(title)


def get_files(data_dir):
    """
    Get L2R train/val files from NLST challenge
    """
    data_json = os.path.join(data_dir, "NLST_dataset.json")

    with open(data_json) as file:
        data = json.load(file)

    train_files = []
    for pair in data["training_paired_images"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        train_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    val_files = []
    for pair in data["registration_val"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        val_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    return train_files, val_files


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from monai.data import Dataset, DataLoader
    from monai.utils import first

    from transforms import get_train_transforms

    root_dir = '../data'
    if root_dir is not None:
        os.makedirs(root_dir, exist_ok=True)
    print(root_dir)

    data_dir = os.path.join(root_dir, "NLST")
    train_files, val_files = get_files(data_dir)
    print(len(train_files), len(val_files))

    print(train_files[0:2])

    full_res_training = False
    if full_res_training:
        target_res = [224, 192, 224]
        spatial_size = [
            -1,
            -1,
            -1,
        ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
    else:
        target_res = [96, 96, 96]
        spatial_size = target_res  # downsample to 96^3 voxels for faster training on resized data (good for testing)

    train_transforms = get_train_transforms(spatial_size, target_res)

    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)

    # Resampled image size
    fixed_image = check_data["fixed_image"][0][0]
    fixed_label = check_data["fixed_label"][0][0]
    moving_image = check_data["moving_image"][0][0]
    moving_label = check_data["moving_label"][0][0]
    print(f"fixed_image shape: {fixed_image.shape}, " f"fixed_label shape: {fixed_label.shape}")
    print(f"moving_image shape: {moving_image.shape}, " f"moving_label shape: {moving_label.shape}")

    # Reorder dims for visualization
    fixed_image = fixed_image.permute(1, 0, 2)
    fixed_label = fixed_label.permute(1, 0, 2)
    moving_image = moving_image.permute(1, 0, 2)
    moving_label = moving_label.permute(1, 0, 2)

    # Image and label visualization
    slice_idx = int(target_res[0] * 95.0 / 224)  # at full-res (224 slices), visualize sagittal slice 95
    fig, axs = plt.subplots(2, 3)
    # plot images
    axs[0, 0].imshow(fixed_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[0, 1].imshow(moving_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 2])
    # plot labels
    axs[1, 0].imshow(fixed_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[1, 1].imshow(moving_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 2])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations")
    plt.tight_layout()
    plt.show()

    # Pointcloud visualization
    fixed_points = check_data["fixed_keypoints"][0]
    moving_points = check_data["moving_keypoints"][0]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2], s=10.0, marker="o", color="lightblue")
    ax.scatter(moving_points[:, 0], moving_points[:, 1], moving_points[:, 2], s=10.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.title("Pointcloud visualizations")
    plt.show()
