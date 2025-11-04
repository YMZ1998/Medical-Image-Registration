import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from monai.networks.blocks import Warp

from parse_args import parse_args, get_net
from utils import load_best_model, to_numpy


class ModelWithWarp(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.warp = Warp()

    def forward(self, inputs):
        ddf = self.model(inputs[:, 0:2])
        moved = self.warp(inputs[:, 2:3], ddf)
        return moved, ddf


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
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"}
        # },
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

    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-02, atol=1e-02)
    print("ONNXRuntime output matches PyTorch output!")


def main():
    warnings.filterwarnings("ignore")

    args = parse_args()
    device = args.device

    model = get_net(args).to(device)
    model_dir = os.path.join("../models", "nlst", args.arch)
    model = load_best_model(model, model_dir)

    model_with_warp = ModelWithWarp(model).to(device)

    input_shape = (1, 3, 192, 192, 192)
    # input_shape = (1, 3, 224, 192, 224)

    save_path = os.path.join("../results", args.arch, "model_with_warp.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    export_to_onnx(model_with_warp, input_shape, save_path, device)


if __name__ == "__main__":
    main()
