import argparse
import os

import torch
from monai.networks.nets import DynUNet, UNet, UNETR, SwinUNETR, SegResNet


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def gpu_usage(device=0):
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3  # GB
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3  # GB

    print(f'GPU {device} usage:')
    print(f'Allocated (current/max): {allocated:.2f} / {max_allocated:.2f} GB')
    print(f'Reserved: {reserved:.2f} GB')
    print(f'Total:    {total:.2f} GB')


def get_net(args):
    print('★' * 30)
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'image size:{args.image_size}')
    print('★' * 30)
    if args.arch == "dynunet":
        net = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=args.num_classes,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            dropout=0.1,
            norm_name="BATCH",
        )
    elif args.arch == "unet":
        net = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=args.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif args.arch == 'unetr':
        net = UNETR(
            spatial_dims=3,
            in_channels=2,
            out_channels=args.num_classes,
            img_size=args.image_size,
            feature_size=16,
            proj_type="conv",
            norm_name="instance",
            dropout_rate=0.1,
        )
    elif args.arch == 'swin_unetr':
        # !wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
        net = SwinUNETR(
            img_size=args.image_size,
            in_channels=2,
            out_channels=args.num_classes,
            feature_size=48,
            use_checkpoint=True,
        )
    elif args.arch == 'seg_resnet':
        net = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=args.num_classes,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            dropout_prob=0.1,
            norm="BATCH",
        )
    else:
        raise ValueError(f"model_name {args.model_name} not supported")

    return net.to(args.device)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dynunet', help='unet/dynunet/seg_resnet')
    parser.add_argument("--data_path", default=r"D:\Data\MIR\NLST2023", type=str, help="training data folder")
    parser.add_argument("--result_path", default="./results", type=str, help="inference folder")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='write model and tensorboard logs')

    parser.add_argument('--full_res_training', action='store_true', default=False, help='full resolution training')
    # parser.add_argument("--image_size", default=(96, 96, 96), type=tuple, help="image size")
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N", help="number of total epochs to train")
    # parser.add_argument("--device", default="cuda", type=str)

    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    args.image_size = [224, 192, 224] if args.full_res_training else [192, 192, 192]
    # args.batch_size = 1 if args.full_res_training else 4

    args.model_dir = os.path.join(os.getcwd(), "models", "nlst", args.arch)
    os.makedirs(args.model_dir, exist_ok=True)

    args.device = get_device()

    print(args)

    return args


if __name__ == '__main__':
    from torchsummary import summary

    get_device()
    args = parse_args()
    # args.arch = 'dynunet'
    # args.image_size = (256, 256, 32)

    model = get_net(args)

    summary(model, (2, args.image_size[0], args.image_size[1], args.image_size[2]))
