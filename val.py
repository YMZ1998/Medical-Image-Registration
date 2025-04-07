import glob
import os
import warnings
from pprint import pprint

import torch
from monai.data import Dataset, DataLoader
from monai.networks.blocks import Warp
from monai.utils import set_determinism, first

from parse_args import get_device, parse_args, get_net
from utils.dataset import get_files
from utils.transforms import get_val_transforms
from utils.utils import forward
from utils.visualization import visualize_registration
import SimpleITK as sitk
import numpy as np


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

    save_dir = os.path.join("results", args.arch)
    os.makedirs(save_dir, exist_ok=True)

    pred_image_array = pred_image[0].cpu().numpy()
    pred_label_array = pred_label[0].cpu().numpy()

    pred_image_array = pred_image_array[0].transpose(2, 1, 0)
    pred_label_array = pred_label_array[0].transpose(2, 1, 0)
    pred_label_array = np.where(pred_label_array > 0, 1, 0)

    pred_image_itk = sitk.GetImageFromArray(pred_image_array)
    pred_label_itk = sitk.GetImageFromArray(pred_label_array)

    pred_image_itk=sitk.Cast(pred_image_itk, sitk.sitkFloat32)
    pred_label_itk=sitk.Cast(pred_label_itk, sitk.sitkUInt8)

    sitk.WriteImage(pred_image_itk, os.path.join(save_dir, "pred_image.nii.gz"))
    sitk.WriteImage(pred_label_itk, os.path.join(save_dir, "pred_label.nii.gz"))

    torch.save(ddf_image[0].cpu(), os.path.join(save_dir, "ddf_image.pt"))
    torch.save(ddf_keypoints[0].cpu(), os.path.join(save_dir, "ddf_keypoints.pt"))
    torch.save(check_data["fixed_image"][0].cpu(), os.path.join(save_dir, "fixed_image.pt"))
    torch.save(check_data["moving_image"][0].cpu(), os.path.join(save_dir, "moving_image.pt"))
    torch.save(check_data["moving_label"][0].cpu(), os.path.join(save_dir, "moving_label.pt"))

    # Visualization
    visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res)


if __name__ == "__main__":
    val()
