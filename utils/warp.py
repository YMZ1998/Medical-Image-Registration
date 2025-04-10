import os

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F


class Warp(nn.Module):
    def __init__(self, mode="bilinear", padding_mode="border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (B, C, D, H, W)
            ddf: Tensor of shape (B, 3, D, H, W) — displacements in (z, y, x)

        Returns:
            Warped image of same shape.
        """
        B, C, D, H, W = image.shape
        assert ddf.shape == (B, 3, D, H, W), f"Expected DDF shape (B, 3, D, H, W), got {ddf.shape}"

        # Build reference grid in voxel space
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, D, device=image.device),
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing="ij"
        )  # each: (D, H, W)

        grid = torch.stack((xx, yy, zz), dim=-1)  # (D, H, W, 3)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, D, H, W, 3)

        # Normalize DDF to [-1, 1]
        ddf_x = ddf[:, 2] * 2 / (W - 1)
        ddf_y = ddf[:, 1] * 2 / (H - 1)
        ddf_z = ddf[:, 0] * 2 / (D - 1)
        ddf_norm = torch.stack((ddf_x, ddf_y, ddf_z), dim=-1)  # (B, D, H, W, 3)

        # Add normalized DDF to grid
        warped_grid = grid + ddf_norm  # (B, D, H, W, 3)

        # Perform warping
        warped = F.grid_sample(
            image,
            warped_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True
        )
        return warped


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    monai_available = True
    try:
        from monai.networks.blocks import Warp as MonaiWarp

        monai_warp = MonaiWarp()
        monai_available = True
    except ImportError:
        print("MONAI not available, skipping comparison")

    # 输入图像 & DDF
    img = torch.randn(1, 1, 96, 96, 96).cuda()
    ddf = torch.randn(1, 3, 96, 96, 96).cuda() * 5

    torch_warp = Warp().cuda().eval()

    # 与 MONAI Warp 进行对比
    if monai_available:
        with torch.no_grad():
            out_monai = monai_warp(img, ddf)
            out_custom = torch_warp(img, ddf)
        print("Max difference:", (out_monai - out_custom).abs().max().item())

    # 导出到 ONNX
    save_path = "warp3d.onnx"
    torch.onnx.export(
        torch_warp,
        (img, ddf),
        save_path,
        input_names=["image", "ddf"],
        output_names=["warped"],
        opset_version=20,
        dynamic_axes={"image": {0: "batch"}, "ddf": {0: "batch"}, "warped": {0: "batch"}},
        verbose=False
    )
    print(f"ONNX model exported to: {save_path}")

    # 验证 ONNX 模型结构
    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")

    # 用 ONNX Runtime 推理并验证
    ort_session = onnxruntime.InferenceSession(save_path)

    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(img),
        ort_session.get_inputs()[1].name: to_numpy(ddf),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    # PyTorch 推理结果
    torch_out = torch_warp(img, ddf)

    # 对比 ONNX 与 PyTorch 输出
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-02, atol=1e-04)
    print("ONNXRuntime output matches PyTorch output!")
    os.remove(save_path)
