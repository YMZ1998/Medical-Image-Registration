import torch
from monai.networks import convert_to_onnx
from monai.networks.nets import UNet, SegResNet
import onnxruntime as ort

def test_seg_resnet_export():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SegResNet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=3,
    #     blocks_down=(1, 2, 2, 4),
    #     blocks_up=(1, 1, 1),
    #     init_filters=16,
    #     dropout_prob=0.2,
    # ).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # input_tensor = torch.randn((1, 2, 96, 96, 96), device=device)
    input_tensor = torch.randn((1, 2, 192, 192, 192), device=device)
    # input_tensor = torch.randn((1, 2, 224, 192, 224), device=device)
    onnx_model = convert_to_onnx(
        model=model,
        inputs=[input_tensor],
        input_names=["x"],
        output_names=["y"],
        verify=True,
        device=device,
        use_trace=True,
        use_ort=True,
        opset_version=20,
        rtol=1e-3,
        atol=1e-5,
    )
    print("SegResNet export successful")


if __name__ == "__main__":
    test_seg_resnet_export()
