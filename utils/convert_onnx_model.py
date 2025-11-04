import os
import warnings
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn

from parse_args import parse_args, get_net
from utils import to_numpy, load_best_model


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        ddf = self.model(inputs)
        return ddf


def export_to_onnx(model, input_shape, save_path="model.onnx", device="cuda"):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    torch_output = model(dummy_input)
    # with torch.no_grad():
    #     torch_output = model(dummy_input)
    #     if isinstance(torch_output, (list, tuple)):
    #         torch_output = torch_output[0]
    #     elif isinstance(torch_output, dict):
    #         torch_output = next(iter(torch_output.values()))

    print(f"Output shape: {torch_output.shape}")

    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=20,
        # dynamic_axes=None,  # 医学影像尺寸固定
        verbose=False,
    )

    print(f"ONNX model saved to: {save_path}")

    # 验证模型结构
    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except Exception as e:
        print(f"ONNX model check failed: {e}")
        return

    # 推理一致性验证
    ort_session = onnxruntime.InferenceSession(save_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_output), ort_outputs[0], rtol=1e-2, atol=1e-2)
    print("ONNXRuntime output matches PyTorch output.")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    device = 'cpu'

    model = get_net(args).to(device)
    model_dir = os.path.join("../models", "nlst", args.arch)
    model = load_best_model(model, model_dir)

    model_ddf = Model(model).to(device)

    input_shape = (1, 2, 192, 192, 192)
    save_path = os.path.join("../results", args.arch.replace("/", "_"), "model.onnx")

    export_to_onnx(model_ddf, input_shape, save_path, device)


if __name__ == "__main__":
    main()
