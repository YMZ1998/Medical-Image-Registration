import os
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
from monai.networks.blocks import Warp

from parse_args import parse_args


# from warp import Warp


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_to_onnx(model, input_shapes, save_path="model.onnx", device="cuda"):
    model.eval()

    # 创建多个输入张量
    dummy_input1 = torch.randn(*input_shapes[0]).to(device)
    dummy_input2 = torch.randn(*input_shapes[1]).to(device)

    torch_out = model(dummy_input1, dummy_input2)  # 修改以适应多个输入

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

    model_with_warp = Warp().to(device)

    # 设置输入 shape
    input_shape = [(1, 1, args.image_size[0], args.image_size[1], args.image_size[2]),
                   (1, 3, args.image_size[0], args.image_size[1], args.image_size[2])]

    # input_shape = [(1, 1, 96, 96, 96),
    #                (1, 3, 96, 96, 96)]
    # 导出路径
    save_path = os.path.join("../results", args.arch, "warp.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 导出并验证
    export_to_onnx(model_with_warp, input_shape, save_path, device)


if __name__ == "__main__":
    main()
