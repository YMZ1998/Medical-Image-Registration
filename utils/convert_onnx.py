import glob
import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from monai.networks import convert_to_onnx

from parse_args import parse_args, get_net
from monai.networks.blocks import Warp
# from warp import Warp


class ModelWithWarp(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.warp = Warp()

    def forward(self, inputs):
        ddf = self.model(inputs[:, 0:2])
        moved = self.warp(inputs[:, 2:3], ddf)
        return moved, ddf


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_model_to_onnx(model, input_shape, onnx_path="model.onnx", device="cuda"):
    model = model.to(device).eval()

    dummy_input = torch.randn(*input_shape, requires_grad=False).to(device)

    onnx_model = convert_to_onnx(
        model=model,
        inputs=[dummy_input],
        input_names=["input"],
        output_names=["output"],
        verify=True,  # 验证 PyTorch 和 ONNX 输出是否一致
        device=device,
        rtol=1e-02, atol=1e-04,
        use_trace=True,  # True 使用 torch.jit.trace，False 使用 torch.onnx.export
        use_ort=True,  # 使用 ONNX Runtime 进行验证
        opset_version=20  # 建议 16~18，20 太新很多库还不支持
    )
    print(f"Exported ONNX model saved to {onnx_path}")
    return onnx_model


def export_to_onnx(model, input_shapes, save_path="model.onnx", device="cuda"):
    model.eval()

    dummy_input = torch.randn(*input_shapes).to(device)

    torch_out = model(dummy_input)

    print(torch_out[0].shape)

    torch.onnx.export(
        model,
        (dummy_input),
        save_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=20,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        verbose=False,
    )
    print(f"ONNX model exported to: {save_path}")

    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return

    ort_session = onnxruntime.InferenceSession(save_path)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(dummy_input),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-02, atol=1e-04)
    print("ONNXRuntime output matches PyTorch output!")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    device = args.device

    model = get_net(args).to(device)
    args.model_dir = os.path.join("../models", "nlst", args.arch)
    best_model_files = glob.glob(os.path.join(args.model_dir, "*_kpt_loss_best_tre*"))

    if not best_model_files:
        raise FileNotFoundError("No best model checkpoint found!")

    print(f"Loading weights from: {best_model_files[0]}")
    model.load_state_dict(torch.load(best_model_files[0], weights_only=False, map_location='cpu'), strict=False)

    model_with_warp = ModelWithWarp(model).to(device)

    input_shape = (1, 3, 192, 192, 192)
    # input_shape = (1, 3, 224, 192, 224)

    save_path = os.path.join("../results", args.arch, "model_with_warp.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    export_to_onnx(model_with_warp, input_shape, save_path, device)


if __name__ == "__main__":
    main()
