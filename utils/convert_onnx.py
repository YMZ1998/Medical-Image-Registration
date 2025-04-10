import glob
import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
# from monai.networks.blocks import Warp
from warp import Warp

from parse_args import parse_args, get_net


class ModelWithWarp(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.warp = Warp()

    def forward(self, inputs):
        # 假设 inputs shape: [B, 3, D, H, W]
        moving = inputs[:, 0:1]           # [B, 1, D, H, W]
        fixed = inputs[:, 1:2]            # [B, 1, D, H, W]
        original_moving = inputs[:, 2:3]  # [B, 1, D, H, W]

        x = torch.cat((moving, fixed), dim=1)  # [B, 2, D, H, W]
        ddf = self.model(x)
        moved = self.warp(original_moving, ddf)
        return moved



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_to_onnx(model, input_shapes, save_path="model.onnx", device="cuda"):
    model.eval()

    dummy_input1 = torch.randn(*input_shapes).to(device)

    torch_out = model(dummy_input1)

    # import SimpleITK as sitk
    # pred_image_array = torch_out[0].cpu().detach().numpy()[0].transpose(2, 1, 0)
    # pred_image_itk = sitk.Cast(sitk.GetImageFromArray(pred_image_array), sitk.sitkFloat32)
    # sitk.WriteImage(pred_image_itk, os.path.join('../results', "torch.nii.gz"))

    print(torch_out.shape)

    torch.onnx.export(
        model,
        (dummy_input1),  # 传递多个输入
        save_path,
        input_names=["input1"],  # 指定多个输入的名字
        output_names=["output"],
        export_params=True,
        opset_version=20,
        dynamic_axes={
            "input1": {0: "batch_size"},

            "output": {0: "batch_size"}
        },
        verbose=False,
    )
    print(f"ONNX model exported to: {save_path}")

    # 验证和比较 ONNX 模型
    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return

    # 使用 ONNXRuntime 进行推理并验证
    ort_session = onnxruntime.InferenceSession(save_path)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(dummy_input1),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-02, atol=1e-04)
    print("ONNXRuntime output matches PyTorch output!")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    device = args.device

    # 获取网络结构并加载模型
    model = get_net(args).to(device)
    args.model_dir = os.path.join("../models", "nlst", args.arch)
    best_model_files = glob.glob(os.path.join(args.model_dir, "*_kpt_loss_best_tre*"))

    if not best_model_files:
        raise FileNotFoundError("No best model checkpoint found!")

    print(f"Loading weights from: {best_model_files[0]}")
    # model.load_state_dict(torch.load(best_model_files[0], weights_only=False, map_location='cpu'), strict=False)

    model_with_warp = ModelWithWarp(model).to(device)

    # 设置输入 shape
    # input_shape = (1, 1, args.image_size[0], args.image_size[1], args.image_size[2])

    input_shape = (1, 3, 96, 96, 96)
    # input_shape = (1, 3, 224, 192, 224)
    # 导出路径
    save_path = os.path.join("../results", args.arch, "model_with_warp.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 导出并验证
    export_to_onnx(model_with_warp, input_shape, save_path, device)


if __name__ == "__main__":
    main()
