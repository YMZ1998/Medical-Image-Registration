import os
import shutil
import warnings
from pprint import pprint

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort
import torch
from monai.transforms import Compose, LoadImage, Resize, ToTensor, ScaleIntensityRange
from monai.utils import set_determinism

from parse_args import parse_args
from utils.dataset import get_test_files
from utils.utils import remove_and_create_dir
from utils.visualization import visualize_one_case


def get_transforms(spatial_size, normalize=True):
    transforms = [
        LoadImage(image_only=True, ensure_channel_first=True),
        Resize(spatial_size, mode="trilinear", align_corners=True),
        ToTensor()
    ]
    if normalize:
        transforms.insert(1, ScaleIntensityRange(a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True))
    return Compose(transforms)


def load_image(image_path, spatial_size, normalize=True):
    return get_transforms(spatial_size, normalize)(image_path).unsqueeze(0)


def save_array_as_nii(array, file_path, reference=None):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def predict_single_onnx():
    set_determinism(seed=0)
    warnings.filterwarnings("ignore")

    args = parse_args()
    spatial_size = args.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image paths
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))
    case_id = 5
    pprint(test_files[case_id])
    fixed_path = test_files[case_id]["fixed_image"]
    moving_path = test_files[case_id]["moving_image"]

    # Preprocess
    fixed = load_image(fixed_path, spatial_size)
    moving = load_image(moving_path, spatial_size)
    original_moving = load_image(moving_path, spatial_size, normalize=False)

    input_tensor = torch.cat((moving, fixed, original_moving), dim=1).numpy().astype(np.float32)

    print(f"Input tensor shape: {input_tensor.shape}")

    # Load ONNX model
    onnx_path = os.path.join("./results", args.arch, "model_with_warp.onnx")
    ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Inference
    ort_inputs = {"input": input_tensor}
    moved_np, ddf_np = ort_session.run(None, ort_inputs)

    moved = torch.tensor(moved_np).to(device)
    ddf = torch.tensor(ddf_np).to(device)

    print(f"Moved shape: {moved.shape}, DDF shape: {ddf.shape}")

    # Visualize
    print("Visualizing...")
    visualize_one_case({"fixed_image": fixed, "moving_image": moving}, moved, ddf)

    # Save results
    print("Saving results...")
    save_dir = os.path.join("results", args.arch)
    # remove_and_create_dir(save_dir)

    pred_array = moved[0].cpu().numpy()[0].transpose(2, 1, 0)
    if args.full_res_training:
        save_array_as_nii(pred_array, os.path.join(save_dir, "pred_image.nii.gz"), reference=sitk.ReadImage(fixed_path))
        shutil.copy(fixed_path, os.path.join(save_dir, "fixed_image.nii.gz"))
        shutil.copy(moving_path, os.path.join(save_dir, "moving_image.nii.gz"))
    else:
        pred_itk = sitk.Cast(sitk.GetImageFromArray(pred_array), sitk.sitkFloat32)
        sitk.WriteImage(pred_itk, os.path.join(save_dir, "pred_image.nii.gz"))

    print("ONNX inference done!")


if __name__ == "__main__":
    predict_single_onnx()
