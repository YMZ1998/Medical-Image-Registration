import glob
import os
import warnings
from pprint import pprint

import SimpleITK as sitk
import torch
from monai.networks.blocks import Warp
from monai.transforms import (
    LoadImage, Compose, Resize, ToTensor, ScaleIntensityRange
)
from monai.utils import set_determinism

from parse_args import parse_args, get_net
from utils.dataset import get_test_files
from utils.utils import forward
from utils.visualization import visualize_one_case


def get_infer_transforms(spatial_size):
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensityRange(a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size, mode="trilinear", align_corners=True),
        ToTensor()
    ])


def load_and_preprocess(image_path, spatial_size):
    transforms = get_infer_transforms(spatial_size)
    return transforms(image_path).unsqueeze(0)


def predict_single():
    # Setup
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    args = parse_args()
    target_res = [224, 192, 224] if args.full_res_training else [96, 96, 96]
    spatial_size = [-1, -1, -1] if args.full_res_training else target_res

    # Load test image pair (first one)
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))

    case_id = 2

    pprint(test_files[case_id])
    fixed_image_path = test_files[case_id]["fixed_image"]
    moving_image_path = test_files[case_id]["moving_image"]

    # Preprocess
    fixed_image = load_and_preprocess(fixed_image_path, spatial_size)
    moving_image = load_and_preprocess(moving_image_path, spatial_size)

    # Load model
    device = args.device
    model = get_net(args).to(device)
    warp_layer = Warp().to(device)

    # Load checkpoint
    best_model_files = glob.glob(os.path.join(args.model_dir, "*_kpt_loss_best_tre*"))
    if not best_model_files:
        raise FileNotFoundError("No best model checkpoint found!")
    model.load_state_dict(torch.load(best_model_files[0], weights_only=True))

    # Inference
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=args.amp):
            ddf_image, _, pred_image, _ = forward(
                fixed_image.to(device),
                moving_image.to(device),
                None,
                None,
                model,
                warp_layer,
            )

    print("Saving results...")
    save_dir = os.path.join("results", args.arch)
    os.makedirs(save_dir, exist_ok=True)

    pred_image_array = pred_image[0].cpu().numpy()[0].transpose(2, 1, 0)
    pred_image_itk = sitk.Cast(sitk.GetImageFromArray(pred_image_array), sitk.sitkFloat32)
    sitk.WriteImage(pred_image_itk, os.path.join(save_dir, "pred_image.nii.gz"))

    save_pt = False
    if save_pt:
        torch.save(ddf_image[0].cpu(), os.path.join(save_dir, "ddf_image.pt"))
        torch.save(fixed_image[0].cpu(), os.path.join(save_dir, "fixed_image.pt"))
        torch.save(moving_image[0].cpu(), os.path.join(save_dir, "moving_image.pt"))

    check_data = {
        "fixed_image": fixed_image,
        "moving_image": moving_image,
    }
    print("Visualizing...")
    visualize_one_case(check_data, pred_image, target_res)


if __name__ == "__main__":
    predict_single()
