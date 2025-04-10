import glob
import os
import shutil
import warnings
import numpy as np
import onnxruntime as ort
from pprint import pprint

import SimpleITK as sitk
import torch

from utils.dataset import get_test_files
from utils.warp import Warp
from monai.transforms import (
    LoadImage, Compose, Resize, ToTensor, ScaleIntensityRange
)
from monai.utils import set_determinism

from parse_args import parse_args
from utils.utils import remove_and_create_dir
from utils.visualization import visualize_one_case


def get_infer_transforms(spatial_size):
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensityRange(a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size, mode="trilinear", align_corners=True),
        ToTensor()
    ])


def get_moving_transforms(spatial_size):
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        Resize(spatial_size, mode="trilinear", align_corners=True),
        ToTensor()
    ])


def load_and_preprocess(image_path, spatial_size):
    transforms = get_infer_transforms(spatial_size)
    return transforms(image_path).unsqueeze(0)


def load_moving(image_path, spatial_size):
    transforms = get_moving_transforms(spatial_size)
    return transforms(image_path).unsqueeze(0)


def save_array_as_nii(array, file_path, reference=None):
    # array = array * 1600 - 1200
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def predict_single_onnx():
    set_determinism(seed=0)
    warnings.filterwarnings("ignore")

    args = parse_args()
    target_res = [224, 192, 224] if args.full_res_training else [192, 192, 192]
    spatial_size = [-1, -1, -1] if args.full_res_training else target_res
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))
    case_id = 5
    pprint(test_files[case_id])
    fixed_image_path = test_files[case_id]["fixed_image"]
    moving_image_path = test_files[case_id]["moving_image"]

    fixed_image = load_and_preprocess(fixed_image_path, spatial_size)
    moving_image = load_and_preprocess(moving_image_path, spatial_size)
    original_moving_image = load_moving(moving_image_path, spatial_size)

    input_tensor = torch.cat((moving_image, fixed_image, original_moving_image), dim=1).numpy().astype(np.float32)

    print(input_tensor.shape)

    # Load ONNX model
    onnx_model_path = os.path.join("./results", args.arch, "model_with_warp.onnx")
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Inference
    ort_inputs = {"input": input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    moved_image = torch.tensor(ort_outs[0]).to(device)
    ddf_image = torch.tensor(ort_outs[1]).to(device)

    print(moved_image.shape, ddf_image.shape)


    check_data = {
        "fixed_image": fixed_image,
        "moving_image": moving_image,
    }
    print("Visualizing...")
    visualize_one_case(check_data, moved_image, ddf_image, target_res)

    print("Saving results...")
    save_dir = os.path.join("results", args.arch)
    remove_and_create_dir(save_dir)

    pred_image_array = moved_image[0].cpu().numpy()[0].transpose(2, 1, 0)
    if args.full_res_training:
        save_array_as_nii(pred_image_array, os.path.join(save_dir, "pred_image.nii.gz"),
                          reference=sitk.ReadImage(fixed_image_path))

        shutil.copy(fixed_image_path, os.path.join(save_dir, "fixed_image.nii.gz"))
        shutil.copy(moving_image_path, os.path.join(save_dir, "moving_image.nii.gz"))
    else:
        pred_image_itk = sitk.Cast(sitk.GetImageFromArray(pred_image_array), sitk.sitkFloat32)
        sitk.WriteImage(pred_image_itk, os.path.join(save_dir, "pred_image.nii.gz"))

    print("ONNX inference done!")


if __name__ == "__main__":
    predict_single_onnx()
