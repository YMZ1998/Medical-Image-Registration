import os
import shutil
import warnings
from pprint import pprint

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort
import torch
from monai.utils import set_determinism

from parse_args import parse_args
from utils.dataset import get_test_files
from utils.infer_transforms import load_image
from utils.process_image import save_array_as_nii
from utils.visualization import visualize_one_case


def predict_single_onnx():
    set_determinism(seed=0)
    warnings.filterwarnings("ignore")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image paths
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))
    case_id = 5
    pprint(test_files[case_id])
    fixed_path = test_files[case_id]["fixed_image"]
    moving_path = test_files[case_id]["moving_image"]

    # Preprocess
    fixed = load_image(fixed_path, args.image_size)
    moving = load_image(moving_path, args.image_size)
    original_moving = load_image(moving_path, args.image_size, normalize=False)

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

    pred_array = moved[0].cpu().numpy()[0].transpose(2, 1, 0)
    save_array_as_nii(pred_array, os.path.join(save_dir, "pred_image.nii.gz"), reference=sitk.ReadImage(fixed_path))
    shutil.copy(fixed_path, os.path.join(save_dir, "fixed_image.nii.gz"))
    shutil.copy(moving_path, os.path.join(save_dir, "moving_image.nii.gz"))


    print("ONNX inference done!")


if __name__ == "__main__":
    predict_single_onnx()
