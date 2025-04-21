import os
import shutil
import warnings
from pprint import pprint

import SimpleITK as sitk
import torch
from monai.utils import set_determinism

from parse_args import parse_args, get_net
from utils import load_best_model
from utils.dataset import get_test_files
from utils.infer_transforms import load_image
from utils.process_image import save_array_as_nii
from utils.visualization import visualize_one_case
from monai.networks.blocks import Warp


# from utils.warp import Warp


def predict_single():
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    args = parse_args()
    spatial_size = [-1, -1, -1] if args.full_res_training else args.image_size

    test_files = get_test_files(os.path.join(args.data_path, "NLST"))

    case_id = 5

    pprint(test_files[case_id])
    fixed_image_path = test_files[case_id]["fixed_image"]
    moving_image_path = test_files[case_id]["moving_image"]

    # Preprocess
    fixed_image = load_image(fixed_image_path, spatial_size)
    moving_image = load_image(moving_image_path, spatial_size)
    original_moving_image = load_image(moving_image_path, spatial_size, normalize=False)

    # Load model
    device = args.device
    model = get_net(args).to(device)
    warp_layer = Warp().to(device)

    # Load checkpoint
    model = load_best_model(model, args.model_dir)

    # Inference
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=args.amp):
            fixed_image = fixed_image.to(device)
            moving_image = moving_image.to(device)
            original_moving_image = original_moving_image.to(device)
            ddf_image = model(torch.cat((moving_image, fixed_image), dim=1)).float()
            # warp moving image and label with the predicted ddf
            # pred_image = warp_layer(moving_image, ddf_image)
            original_pred_image = warp_layer(original_moving_image, ddf_image)

    check_data = {
        "fixed_image": fixed_image,
        "moving_image": moving_image,
    }
    print("Visualizing...")
    visualize_one_case(check_data, original_pred_image, ddf_image)

    print("Saving results...")
    save_dir = os.path.join("results", args.arch)
    print("Saving results to: ", save_dir)

    pred_image_array = original_pred_image[0].cpu().numpy()[0].transpose(2, 1, 0)
    save_array_as_nii(pred_image_array, os.path.join(save_dir, "pred_image.nii.gz"),
                      reference=sitk.ReadImage(fixed_image_path))

    shutil.copy(fixed_image_path, os.path.join(save_dir, "fixed_image.nii.gz"))
    shutil.copy(moving_image_path, os.path.join(save_dir, "moving_image.nii.gz"))

    save_pt = False
    if save_pt:
        torch.save(ddf_image[0].cpu(), os.path.join(save_dir, "ddf_image.pt"))
        torch.save(fixed_image[0].cpu(), os.path.join(save_dir, "fixed_image.pt"))
        torch.save(moving_image[0].cpu(), os.path.join(save_dir, "moving_image.pt"))


if __name__ == "__main__":
    predict_single()
