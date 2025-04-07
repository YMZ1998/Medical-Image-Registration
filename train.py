import glob
import os
import time
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import SegResNet
from monai.utils import set_determinism, first
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from parse_args import get_device, parse_args, get_net
from utils.dataset import get_files, overlay_img
from utils.train_and_eval import train_one_epoch, evaluate_model, save_best_model, save_latest_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.utils import forward, loss_fun, tre, collate_fn, remove_and_create_dir, plot_training_logs

if __name__ == "__main__":
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    # monai.config.print_config()

    args = parse_args()

    train_files, val_files = get_files(os.path.join(args.data_folder, "NLST"))

    pprint(train_files[0:2])

    full_res_training = False
    if full_res_training:
        target_res = [224, 192, 224]
        # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
        spatial_size = [-1, -1, -1]
    else:
        target_res = [96, 96, 96]
        # downsample to 96^3 voxels for faster training on resized data (good for testing)
        spatial_size = target_res

    train_transforms = get_train_transforms(spatial_size, target_res)
    val_transforms = get_val_transforms(spatial_size)

    device = get_device()
    # image voxel size at target resolution
    vx = np.array([1.5, 1.5, 1.5]) / (np.array(target_res) / np.array([224, 192, 224]))
    vx = torch.tensor(vx).to(device)

    # tensorboard --logdir="./models/nlst/seg_resnet"
    dir_save = os.path.join(os.getcwd(), "models", "nlst", args.arch)
    # remove_and_create_dir(dir_save)
    os.makedirs(dir_save, exist_ok=True)
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=dir_save)

    print("Prepare dataset...")
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model
    model = get_net(args)
    warp_layer = Warp().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Metrics
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # # Automatic mixed precision (AMP) for faster training
    # scaler = torch.GradScaler("cuda")
    #
    # # Start torch training loop
    best_eval_tre = float("inf")
    best_eval_dice = 0
    log_train_loss = []
    log_val_dice = []
    log_val_tre = []
    pth_best_tre, pth_best_dice, pth_latest = "", "", ""

    for epoch in range(args.epochs):
        # Train
        epoch_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_fun, warp_layer, device, args,
                                     writer)
        log_train_loss.append(epoch_loss)

        # Eval
        tre_after, dice_after = evaluate_model(model, warp_layer, val_loader, device, args, vx, writer)

        # === Save best models ===
        pth_best_tre, best_eval_tre = save_best_model(
            model, epoch, tre_after, best_eval_tre, args.arch, "tre", dir_save, pth_best_tre
        )

        pth_best_dice, best_eval_dice = save_best_model(
            model, epoch, dice_after, best_eval_dice, args.arch, "dice", dir_save, pth_best_dice
        )

        # === Save latest ===
        pth_latest = save_latest_model(model, args.arch, dir_save, pth_latest)

    plot_training_logs(
        logs=[log_train_loss, log_val_dice, log_val_tre],
        titles=["Train Loss", "Validation Dice", "Validation TRE"],
        save_path=os.path.join(dir_save, "training_logs.png")
    )

    load_pretrained_model_weights = False
    if load_pretrained_model_weights:
        dir_load = dir_save  # folder where network weights are stored
        # instantiate warp layer
        warp_layer = Warp().to(device)
        # instantiate model
        model = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            dropout_prob=0.2,
        )
        # load model weights
        filename_best_model = glob.glob(os.path.join(dir_load, f"{args.arch}_kpt_loss_best_tre*"))[0]
        model.load_state_dict(torch.load(filename_best_model, weights_only=True))
        # to GPU
        model.to(device)

    set_determinism(seed=1)
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)

    # Forward pass
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=args.amp):
            ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                check_data["fixed_image"].to(device),
                check_data["moving_image"].to(device),
                check_data["moving_label"].to(device),
                check_data["fixed_keypoints"].to(device),
                model,
                warp_layer,
            )

    # Image and label visualization
    fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
    fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
    moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
    moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()
    # choose slice
    slice_idx = int(target_res[0] * 95.0 / 224)  # visualize slice 95 at full-res (224 slices)
    # plot images
    fig, axs = plt.subplots(2, 2)
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 0], "Before registration")
    overlay_img(fixed_image, pred_image, slice_idx, axs[0, 1], "After registration")
    # plot labels
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 0])
    overlay_img(fixed_label, pred_label, slice_idx, axs[1, 1])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations pre-/post-registration")
    plt.tight_layout()
    plt.show()

    # Pointcloud visualization
    fixed_keypoints = check_data["fixed_keypoints"][0].cpu()
    moving_keypoints = check_data["moving_keypoints"][0].cpu()
    moved_keypoints = fixed_keypoints + ddf_keypoints[0].cpu()
    # plot pointclouds
    fig = plt.figure()
    # Before registration
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(fixed_keypoints[:, 0], fixed_keypoints[:, 1], fixed_keypoints[:, 2], s=2.0, marker="o",
               color="lightblue")
    ax.scatter(moving_keypoints[:, 0], moving_keypoints[:, 1], moving_keypoints[:, 2], s=2.0, marker="o",
               color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    # After registration
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(moved_keypoints[:, 0], moved_keypoints[:, 1], moved_keypoints[:, 2], s=2.0, marker="o",
               color="lightblue")
    ax.scatter(moving_keypoints[:, 0], moving_keypoints[:, 1], moving_keypoints[:, 2], s=2.0, marker="o",
               color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.show()
