import json
import os


def build_file_entry(data_dir, pair, only_image=False):
    fixed = os.path.basename(pair["fixed"]).split(".")[0]
    moving = os.path.basename(pair["moving"]).split(".")[0]
    if only_image:
        return {
            "fixed_image": os.path.join(data_dir, "imagesTr", f"{fixed}.nii.gz"),
            "moving_image": os.path.join(data_dir, "imagesTr", f"{moving}.nii.gz"),
        }
    else:
        return {
            "fixed_image": os.path.join(data_dir, "imagesTr", f"{fixed}.nii.gz"),
            "moving_image": os.path.join(data_dir, "imagesTr", f"{moving}.nii.gz"),
            "fixed_label": os.path.join(data_dir, "masksTr", f"{fixed}.nii.gz"),
            "moving_label": os.path.join(data_dir, "masksTr", f"{moving}.nii.gz"),
            "fixed_keypoints": os.path.join(data_dir, "keypointsTr", f"{fixed}.csv"),
            "moving_keypoints": os.path.join(data_dir, "keypointsTr", f"{moving}.csv"),
        }


def get_files(data_dir):
    data_json = os.path.join(data_dir, "NLST_dataset.json")
    with open(data_json, "r") as f:
        data = json.load(f)
    train_files = [build_file_entry(data_dir, pair) for pair in data["training_paired_images"]]
    val_files = [build_file_entry(data_dir, pair)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           for pair in data["registration_val"]]

    return train_files, val_files


def get_test_files(data_dir):
    data_json = os.path.join(data_dir, "NLST_dataset.json")
    with open(data_json, "r") as f:
        data = json.load(f)

    test_files = [build_file_entry(data_dir, pair, True) for pair in data["registration_val"]]

    return test_files


# -------------------- TESTING CODE BELOW -------------------- #

def visualize_check_sample(train_files, spatial_size, target_res):
    from matplotlib import pyplot as plt
    from monai.data import Dataset, DataLoader
    from monai.utils import first
    from transforms import get_train_transforms
    from visualization import overlay_img

    train_transforms = get_train_transforms(spatial_size, target_res)
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)

    # Extract and permute images for visualization
    def prep(vol): return vol[0][0].permute(1, 0, 2)

    fixed_image = prep(check_data["fixed_image"])
    moving_image = prep(check_data["moving_image"])
    fixed_label = prep(check_data["fixed_label"])
    moving_label = prep(check_data["moving_label"])

    slice_idx = int(target_res[0] * 95.0 / 224)

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(fixed_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[0, 1].imshow(moving_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 2])

    axs[1, 0].imshow(fixed_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[1, 1].imshow(moving_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 2])

    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and Label Visualization")
    plt.tight_layout()
    plt.show()

    # Visualize keypoints
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(check_data["fixed_keypoints"][0][:, 0], check_data["fixed_keypoints"][0][:, 1],
               check_data["fixed_keypoints"][0][:, 2],
               s=10.0, marker="o", color="lightblue")
    ax.scatter(check_data["moving_keypoints"][0][:, 0], check_data["moving_keypoints"][0][:, 1],
               check_data["moving_keypoints"][0][:, 2],
               s=10.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.title("Pointcloud Visualization")
    plt.show()


if __name__ == "__main__":
    root_dir = "../data"
    os.makedirs(root_dir, exist_ok=True)
    data_dir = os.path.join(root_dir, "NLST")

    train_files, val_files = get_files(data_dir)
    print(f"# Training pairs: {len(train_files)}, # Validation pairs: {len(val_files)}")
    print(train_files[:2])

    # Set resolution
    full_res_training = False
    target_res = [224, 192, 224] if full_res_training else [96, 96, 96]
    spatial_size = [-1, -1, -1] if full_res_training else target_res

    visualize_check_sample(train_files, spatial_size, target_res)



