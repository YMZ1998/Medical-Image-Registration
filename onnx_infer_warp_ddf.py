import os
import shutil
import warnings

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort
import torch
from monai.utils import set_determinism

from parse_args import parse_args
from utils.dataset import get_test_files
from utils.infer_transforms import load_image
from utils.visualization import visualize_one_case


def warp_image_with_ddf(original_image, ddf, reference):
    ddf = ddf.astype(np.float64)

    if np.all(ddf == 0):
        print("Warning: The displacement field (ddf) is all zeros. Please check the source of ddf.")
    else:
        print(f"DDF shape: {ddf.shape}")

    # 确保 DDF 的尺寸与 reference 图像的尺寸一致
    if ddf.shape[1:] != reference.GetSize():  # SimpleITK 的尺寸是 Z, Y, X，需要翻转
        print("DDF size doesn't match reference image size. Resizing DDF...")
        # 将 DDF 转换为 SimpleITK 图像（是向量图像）
        ddf_resampled = sitk.GetImageFromArray(ddf.transpose(3, 2, 1, 0), isVector=True)
        ddf_resampled.SetSpacing(reference.GetSpacing())
        ddf_resampled.SetOrigin(reference.GetOrigin())
        ddf_resampled.SetDirection(reference.GetDirection())

        # 使用 ResampleImageFilter 进行重采样，保持向量的每个分量
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(reference.GetSize())  # 将 DDF 的大小调整为与参考图像相同
        resampler.SetInterpolator(sitk.sitkLinear)  # 使用线性插值
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())

        # 执行重采样
        resampled_ddf_field = resampler.Execute(ddf_resampled)
        # 将调整后的 DDF 用于变形
        ddf_field = sitk.Cast(resampled_ddf_field, sitk.sitkVectorFloat64)
    else:
        # 如果 DDF 大小已经匹配，则直接转换为位移场
        ddf_field = sitk.GetImageFromArray(ddf.transpose(3, 2, 1, 0), isVector=True)
        ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    # 复制空间信息，但不改变数据类型
    ddf_field.CopyInformation(reference)  # Ensure DDF field has the same spatial information

    # 可选：保存 DDF 进行调试或可视化
    print("Saving DDF field...")
    sitk.WriteImage(ddf_field, os.path.join("./results", "ddf_field.nii.gz"))

    # 创建 ResampleImageFilter 来执行图像变形
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)  # 使用原图作为参考
    resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值
    resampler.SetDefaultPixelValue(0)  # 默认像素值为 0
    resampler.SetTransform(sitk.DisplacementFieldTransform(ddf_field))  # 使用 DDF 变换

    # 执行变形并返回变形后的图像
    moved = resampler.Execute(original_image)
    return moved


def predict_single_onnx():
    set_determinism(seed=0)
    warnings.filterwarnings("ignore")

    args = parse_args()
    spatial_size = [-1, -1, -1] if args.full_res_training else args.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    test_files = get_test_files(os.path.join(args.data_path, "NLST"))
    case_id = 5
    fixed_image_path = test_files[case_id]["fixed_image"]
    moving_image_path = test_files[case_id]["moving_image"]

    fixed_image = load_image(fixed_image_path, spatial_size)
    moving_image = load_image(moving_image_path, spatial_size)
    original_moving_image = load_image(moving_image_path, spatial_size, normalize=False)

    input_tensor = torch.cat((moving_image, fixed_image), dim=1).numpy().astype(np.float32)

    print(input_tensor.shape)

    # Load ONNX model
    onnx_model_path = os.path.join("./results", args.arch, "model.onnx")
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Inference
    ort_inputs = {"input": input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    ddf_image = torch.tensor(ort_outs[0]).to(device)

    # Warp using SimpleITK instead of the PyTorch warp layer
    # Load fixed image (reference)
    fixed_itk = sitk.ReadImage(fixed_image_path)

    # ddf: (1, 3, D, H, W) -> (3, D, H, W)
    ddf_np = ort_outs[0][0]

    # Load original moving image (raw, unmodified)
    original_moving_itk = sitk.ReadImage(moving_image_path)

    print("Warping full-resolution image with displacement field...")
    moved_itk = warp_image_with_ddf(original_moving_itk, ddf_np, fixed_itk)

    # Visualize the results
    check_data = {
        "fixed_image": fixed_image,
        "moving_image": moving_image,
    }
    print("Visualizing...")
    visualize_one_case(check_data, original_moving_image, ddf_image)

    print("Saving results...")
    save_dir = os.path.join("results", args.arch)

    # Save warped image
    sitk.WriteImage(moved_itk, os.path.join(save_dir, "pred_image.nii.gz"))
    shutil.copy(fixed_image_path, os.path.join(save_dir, "fixed_image.nii.gz"))
    shutil.copy(moving_image_path, os.path.join(save_dir, "moving_image.nii.gz"))

    print("ONNX inference done!")


if __name__ == "__main__":
    predict_single_onnx()
