import glob
import os
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import torch
from monai.data import Dataset, DataLoader
from monai.networks.blocks import Warp
from monai.utils import set_determinism, first

from parse_args import get_device, parse_args, get_net
from utils.dataset import get_files, overlay_img
from utils.transforms import get_val_transforms
from utils.utils import forward


def visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res):
    # Permute to get correct axis for visualization
    def prep_slice(vol): return vol[0][0].permute(1, 0, 2).cpu()

    fixed_image = prep_slice(check_data["fixed_image"])
    fixed_label = prep_slice(check_data["fixed_label"])
    moving_image = prep_slice(check_data["moving_image"])
    moving_label = prep_slice(check_data["moving_label"])
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()

    slice_idx = int(target_res[0] * 95.0 / 224)  # Slice equivalent to 95 in 224-depth

    fig, axs = plt.subplots(2, 2)
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 0], "Before registration")
    overlay_img(fixed_image, pred_image, slice_idx, axs[0, 1], "After registration")
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 0])
    overlay_img(fixed_label, pred_label, slice_idx, axs[1, 1])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations pre-/post-registration")
    plt.tight_layout()
    plt.show()

    # Keypoint visualization
    fixed_kp = check_data["fixed_keypoints"][0].cpu()
    moving_kp = check_data["moving_keypoints"][0].cpu()
    moved_kp = fixed_kp + ddf_keypoints[0].cpu()

    fig = plt.figure()
    for i, title, fkp in zip(
        [1, 2], ["Before registration", "After registration"], [fixed_kp, moved_kp]
    ):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(fkp[:, 0], fkp[:, 1], fkp[:, 2], s=2.0, color="lightblue")
        ax.scatter(moving_kp[:, 0], moving_kp[:, 1], moving_kp[:, 2], s=2.0, color="orange")
        ax.set_title(title)
        ax.view_init(-10, 80)
        ax.set_aspect("auto")

    plt.show()


def val():
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    args = parse_args()

    # Load data file paths
    train_files, val_files = get_files(os.path.join(args.data_folder, "NLST"))
    pprint(val_files[:2])

    # Resolution setup
    full_res_training = False
    target_res = [224, 192, 224] if full_res_training else [96, 96, 96]
    spatial_size = [-1, -1, -1] if full_res_training else target_res

    val_transforms = get_val_transforms(spatial_size)

    # Prepare device and model
    device = get_device()
    model = get_net(args).to(device)
    warp_layer = Warp().to(device)

    # Load best model checkpoint
    model_dir = os.path.join(os.getcwd(), "models", "nlst", args.arch)
    best_model_files = glob.glob(os.path.join(model_dir, "*_kpt_loss_best_tre*"))
    if not best_model_files:
        raise FileNotFoundError("No best model checkpoint found!")
    model.load_state_dict(torch.load(best_model_files[0], weights_only=True))

    # Validation sample
    set_determinism(seed=1)
    check_loader = DataLoader(Dataset(data=val_files, transform=val_transforms), batch_size=1, shuffle=True)
    check_data = first(check_loader)

    # Inference
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=args.amp):
            ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                check_data["fixed_image"].to(device),
                check_data["moving_image"].to(device),
                check_data["moving_label"].to(device),
                check_data["fixed_keypoints"].to(device),
                model,
                warp_layer,
            )

    # Visualization
    visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res)


if __name__ == "__main__":
    val()
