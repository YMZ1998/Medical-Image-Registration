import argparse
import os
import traceback

import numpy as np
import onnx
import onnxruntime
import torch

from monai.networks.nets import (AHNet, AutoEncoder, BasicUNet, DenseNet,
                                 DynUNet, Generator, HighResNet, SegResNet,
                                 SegResNetVAE, UNet, VarAutoEncoder, VNet,
                                 senet154)

TEST_CASE_1 = [
    UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        num_res_units=2,
        norm="batch"
    ),
    {'original': (1, 1, 160, 160, 160), 'dynamic': (1, 1, 128, 128, 128), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_2 = [
    SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ),
    {'original': (1, 4, 224, 224, 128), 'dynamic': (1, 4, 128, 128, 128), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_3 = [
    AHNet(
        spatial_dims=3,
        psp_block_num=3,
        upsample_mode="nearest",
        out_channels=2,
        pretrained=True
    ),
    {'original': (1, 1, 160, 160, 160), 'dynamic': (1, 1, 128, 128, 128), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_4 = [
    DenseNet(
        init_features=64,
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        spatial_dims=2,
        in_channels=1,
        out_channels=15
    ),
    {'original': (1, 1, 256, 256), 'dynamic': (1, 1, 128, 128), 'spatial_dims': [2, 3]},
]

TEST_CASE_5 = [
    BasicUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        features=[
            32,
            64,
            128,
            256,
            512,
            32
        ]
    ),
    {'original': (1, 3, 128, 128, 128), 'dynamic': (1, 3, 64, 64, 64), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_6 = [
    DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=1
    ),
    {'original': (1, 1, 40, 56, 40), 'dynamic': (1, 1, 40, 40, 40), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_7 = [
    AutoEncoder(spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8), strides=(2, 2)),
    {'original': (2, 1, 32, 32), 'dynamic': (2, 1, 64, 64), 'spatial_dims': [2, 3]},
]

TEST_CASE_8 = [
    Generator(latent_shape=(64,), start_shape=(8, 8, 8), channels=(8, 1), strides=(2, 2), num_res_units=2),
    {'original': (16, 64), 'spatial_dims': [1]},
]

TEST_CASE_9 = [
    HighResNet(spatial_dims=3, in_channels=1, out_channels=3, norm_type="instance"),
    {'original': (1, 1, 32, 24, 48), 'dynamic': (1, 1, 48, 48, 48), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_10 = [
    SegResNetVAE(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
        input_image_size=[224, 224, 128]
    ),
    {'original': (1, 4, 224, 224, 128), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_11 = [
    senet154(spatial_dims=3, in_channels=2, num_classes=2),
    {'original': (2, 2, 64, 64, 64), 'dynamic': (2, 2, 128, 128, 128), 'spatial_dims': [2, 3, 4]},
]

TEST_CASE_12 = [
    VarAutoEncoder(spatial_dims=2, in_shape=(1, 32, 32), out_channels=1, latent_size=2, channels=(4, 8), strides=(2, 2)),
    {'original': (1, 1, 32, 32), 'spatial_dims': [2, 3]},
]

TEST_CASE_13 = [
    VNet(spatial_dims=3, in_channels=1, out_channels=3, dropout_dim=3),
    {'original': (1, 1, 32, 32, 32), 'dynamic': (1, 1, 64, 64, 64), 'spatial_dims': [2, 3, 4]},
]

TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5,
              TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_9, TEST_CASE_10,
              TEST_CASE_11, TEST_CASE_12, TEST_CASE_13]


class StoreShape(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        shape = []
        if not (values[0] == '[' and values[-1] == ']'):
            raise ValueError('Please use correct shape format: [dim1, dim2, ...]')
        for v in values[1:-1].split(","):
            shape.append(int(v))
        setattr(namespace, self.dest, shape)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_dynamic_axes(spatial_dims=None):
    if spatial_dims is None:
        return {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    elif len(spatial_dims) == 2:
        return {'input': {0: 'batch_size', spatial_dims[0]: 'H', spatial_dims[1]: 'W'}, 'output': {0: 'batch_size'}}
    elif len(spatial_dims) == 3:
        return {'input': {0: 'batch_size', spatial_dims[0]: 'H', spatial_dims[1]: 'W', spatial_dims[2]: 'D'},
                'output': {0: 'batch_size'}}
    elif len(spatial_dims) == 1:
        return {'input': {0: 'batch_size', spatial_dims[0]: 'H'}, 'output': {0: 'batch_size'}}
    else:
        raise RuntimeError(f"spatial_dims {spatial_dims} is not supported!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output the TorchScript', default='./models')
    parser.add_argument('--device', type=str, help='device (cpu or cuda:[number])', default='cuda:0')
    parser.add_argument("--export_type", type=str, choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument('--onnx_dynamic_spatial_size', action="store_true")
    args = parser.parse_args()

    passed = []
    for model, input_shape in TEST_CASES:
        test_status = "FAILED"
        print(f"Testing model {model.__class__.__name__}")
        model.to(args.device)
        model.eval()
        dummy_input_1 = torch.randn(input_shape['original'], requires_grad=False).to(args.device)
        torch_out_1 = model(dummy_input_1)

        # if the model does not support dynamic it will not have input_shape['dynamic']
        if input_shape.get('dynamic') is not None:
            dummy_input_2 = torch.randn(input_shape['dynamic'], requires_grad=False).to(args.device)
            torch_out_2 = model(dummy_input_2)
        output_file_prefix = os.path.join(args.output_dir, model.__class__.__name__)
        try:
            if args.export_type == 'torchscript':
                output_file = f"{output_file_prefix}.ts"
                with torch.no_grad():
                    traced_script_module = torch.jit.script(model)
                    traced_script_module.save(output_file)
                test_status = 'EXPORT_PASSED'

                torchscript_model = torch.jit.load(output_file)

                # compare TorchScript and PyTorch results
                torchscript_outs_1 = torchscript_model(dummy_input_1)
                np.testing.assert_allclose(to_numpy(torch_out_1), to_numpy(torchscript_outs_1), rtol=args.rtol,
                                           atol=args.atol)
                print("Exported model has been tested with TorchScript, and the result looks good!")
                test_status = 'NUMERICAL_1_PASSED'

                if input_shape.get('dynamic') is not None:
                    torchscript_outs_2 = torchscript_model(dummy_input_2)
                    np.testing.assert_allclose(to_numpy(torch_out_2), to_numpy(torchscript_outs_2), rtol=args.rtol,
                                               atol=args.atol)
                    print("Exported model has been tested with TorchScript with different input"
                          ", and the result looks good!")
                test_status = ' '
            else:
                # if the model does not support dynamic it will not have input_shape['dynamic']
                if args.onnx_dynamic_spatial_size and input_shape.get('dynamic') is None:
                    passed.append('SKIPPED')
                    continue

                if args.onnx_dynamic_spatial_size:
                    output_file = f"{output_file_prefix}-dynamic.onnx"
                    dynamic_axes = get_dynamic_axes(input_shape['spatial_dims'])
                else:
                    output_file = f"{output_file_prefix}.onnx"
                    dynamic_axes = get_dynamic_axes()

                torch.onnx.export(model,
                                  dummy_input_1,
                                  output_file,
                                  opset_version=11,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes=dynamic_axes)
                test_status = 'EXPORT_PASSED'

                onnx_model = onnx.load(output_file)
                onnx.checker.check_model(onnx_model)
                ort_session = onnxruntime.InferenceSession(output_file)

                ort_inputs_1 = {ort_session.get_inputs()[0].name: to_numpy(dummy_input_1)}
                ort_outs_1 = ort_session.run(None, ort_inputs_1)
                np.testing.assert_allclose(to_numpy(torch_out_1), ort_outs_1[0], rtol=args.rtol, atol=args.atol)
                print("  Exported model has been tested with ONNXRuntime, and the result looks good!")
                test_status = 'NUMERICAL_1_PASSED'

                if args.onnx_dynamic_spatial_size:
                    ort_inputs_2 = {ort_session.get_inputs()[0].name: to_numpy(dummy_input_2)}
                    ort_outs_2 = ort_session.run(None, ort_inputs_2)
                    np.testing.assert_allclose(to_numpy(torch_out_2), ort_outs_2[0], rtol=args.rtol, atol=args.atol)
                    print("  Exported model has been tested with ONNXRuntime with different input, "
                          "and the result looks good!")
                    test_status = 'NUMERICAL_2_PASSED'
                test_status = 'ALL_PASSED'
        except Exception as e:
            print(f"Failed to export {model.__class__.__name__} with {e}")
            print(traceback.format_exc())

        print('=' * 100)
        passed.append(test_status)

    assert len(passed) == len(TEST_CASES), "length of status must match length of TEST CASES"
    print('=' * 100)
    network_header = "Network Name"
    test_status_header = "Status"
    print(f"| {network_header:<20} | {test_status_header:<20} |")
    for idx, test_status in enumerate(passed):
        print(f"| {TEST_CASES[idx][0].__class__.__name__:<20} | {test_status:<20} |")
    print('=' * 100)


if __name__ == "__main__":
    main()