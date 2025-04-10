import glob
import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn

from parse_args import parse_args, get_net


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, moving, fixed):
        x = torch.cat((moving, fixed), dim=1)  # [B, 2, D, H, W]
        ddf = self.model(x)  # 预测位移场
        return ddf


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_to_onnx(model, input_shapes, save_path="model.onnx", device="cuda"):
    model.eval()

    # 创建多个输入张量
    dummy_input1 = torch.randn(*input_shapes).to(device)
    dummy_input2 = torch.randn(*input_shapes).to(device)

    torch_out = model(dummy_input1, dummy_input2)\

    print(torch_out.shape)

    torch.onnx.export(
        model,
        (dummy_input1, dummy_input2),  # 传递多个输入
        save_path,
        input_names=["input1", "input2"],  # 指定多个输入的名字
        output_names=["output"],
        export_params=True,
        opset_version=20,
        dynamic_axes={
            "input1": {0: "batch_size"},
            "input2": {0: "batch_size"},
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
        ort_session.get_inputs()[1].name: to_numpy(dummy_input2),
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
    model.load_state_dict(torch.load(best_model_files[0], weights_only=False, map_location='cpu'), strict=False)

    input_shape = (1, 1, 96, 96, 96)
    # input_shape = (1, 1, 224, 192, 224)

    save_path = os.path.join("../results", args.arch, "model.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model2= Model(model).to(device)
    # 导出并验证
    export_to_onnx(model2, input_shape, save_path, device)


if __name__ == "__main__":
    main()
