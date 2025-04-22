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
from utils.warp import Warp


def predict_single_onnx():
    set_determinism(seed=0)
    warnings.filterwarnings("ignore")

    args = parse_args()
    spatial_size = [-1, -1, -1] if args.full_res_training else args.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))
    case_id = 5
    pprint(test_files[case_id])
    fixed_image_path = test_files[case_id]["fixed_image"]
    moving_image_path = test_files[case_id]["moving_image"]

    fixed_image = load_image(fixed_image_path, spatial_size)
    moving_image = load_image(moving_image_path, spatial_size)
    original_moving_image = load_image(moving_image_path, spatial_size, normalize=False)

    input_tensor = torch.cat((moving_image, fixed_image), dim=1).numpy().astype(np.float32)

    print(input_tensor.shape)

    # Load ONNX model
    onnx_model_path = os.path.join("./results", args.arch, "model.onnx")
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # Inference
    ort_inputs = {"input": input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    ddf_image = torch.tensor(ort_outs[0]).to(device)

    # Warp
    warp_layer = Warp().to(device)
    with torch.no_grad():
        pred_image = warp_layer(moving_image.to(device), ddf_image)
        original_pred_image = warp_layer(original_moving_image.to(device), ddf_image)

    check_data = {
        "fixed_image": fixed_image,
        "moving_image": moving_image,
    }
    print("Visualizing...")
    visualize_one_case(check_data, original_pred_image, ddf_image)

    print("Saving results...")
    save_dir = os.path.join("results", args.arch)

    pred_image_array = pred_image[0].cpu().numpy()[0].transpose(2, 1, 0)
    save_array_as_nii(pred_image_array, os.path.join(save_dir, "pred_image.nii.gz"),
                      reference=sitk.ReadImage(fixed_image_path))

    shutil.copy(fixed_image_path, os.path.join(save_dir, "fixed_image.nii.gz"))
    shutil.copy(moving_image_path, os.path.join(save_dir, "moving_image.nii.gz"))

    print("ONNX inference done!")


if __name__ == "__main__":
    predict_single_onnx()
