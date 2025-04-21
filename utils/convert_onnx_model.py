import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch

from parse_args import parse_args, get_net
from utils.utils import to_numpy, load_best_model


def export_to_onnx(model, input_shape, save_path="model.onnx", device="cuda"):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        torch_output = model(dummy_input)

    print(f"Output shape: {torch_output.shape}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=16,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )

    print(f"ONNX model saved to: {save_path}")

    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except Exception as e:
        print(f"ONNX model check failed: {e}")
        return

    ort_session = onnxruntime.InferenceSession(save_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_output), ort_outputs[0], rtol=1e-2, atol=1e-4)
    print("ONNXRuntime output matches PyTorch output.")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    device = args.device

    model = get_net(args).to(device)
    model_dir = os.path.join("../models", "nlst", args.arch)
    model = load_best_model(model, model_dir)

    input_shape = (1, 2, 192, 192, 192)
    save_path = os.path.join("../results", args.arch, "model.onnx")

    export_to_onnx(model, input_shape, save_path, device)


if __name__ == "__main__":
    main()
