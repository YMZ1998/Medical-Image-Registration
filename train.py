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

from utils.dataset import get_files, overlay_img
from utils.transforms import get_train_transforms, get_val_transforms
from utils.utils import forward, loss_fun, tre, collate_fn

if __name__=="__main__":
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    monai.config.print_config()

    root_dir = './data'
    if root_dir is not None:
        os.makedirs(root_dir, exist_ok=True)
    print(root_dir)

    data_dir = os.path.join(root_dir, "NLST")
    train_files, val_files = get_files(data_dir)

    pprint(train_files[0:2])

    full_res_training = False
    if full_res_training:
        target_res = [224, 192, 224]
        spatial_size = [
            -1,
            -1,
            -1,
        ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
    else:
        target_res = [96, 96, 96]
        spatial_size = target_res  # downsample to 96^3 voxels for faster training on resized data (good for testing)


    train_transforms = get_train_transforms(spatial_size, target_res)
    val_transforms = get_val_transforms(spatial_size)

    # device, optimizer, epoch and batch settings
    device = "cuda:0"
    batch_size = 4
    lr = 1e-4
    weight_decay = 1e-5
    max_epochs = 200

    # image voxel size at target resolution
    vx = np.array([1.5, 1.5, 1.5]) / (np.array(target_res) / np.array([224, 192, 224]))
    vx = torch.tensor(vx).to(device)

    # Use mixed precision feature of GPUs for faster training
    amp_enabled = True

    # loss weights (set to zero to disable loss term)
    lam_t = 1e0  # TRE  (keypoint loss)
    lam_l = 0  # Dice (mask overlay)
    lam_m = 0  # MSE (image similarity)
    lam_r = 0  # Bending loss (smoothness of the DDF)

    #  Write model and tensorboard logs?
    do_save = True
    dir_save = os.path.join(os.getcwd(), "models", "nlst", "tre-segresnet")
    if do_save and not os.path.exists(dir_save):
        os.makedirs(dir_save)

    print("Prepare dataset...")
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # DataLoaders, now with custom function `collate_fn`, to rectify the ragged keypoint tensors
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model
    model = SegResNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        dropout_prob=0.2,
    ).to(device)
    warp_layer = Warp().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Metrics
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Automatic mixed precision (AMP) for faster training
    amp_enabled = True
    scaler = torch.GradScaler("cuda")

    # Tensorboard
    if do_save:
        writer = SummaryWriter(log_dir=dir_save)

    # Start torch training loop
    val_interval = 5
    best_eval_tre = float("inf")
    best_eval_dice = 0
    log_train_loss = []
    log_val_dice = []
    log_val_tre = []
    pth_best_tre, pth_best_dice, pth_latest = "", "", ""
    for epoch in range(max_epochs):
        # ==============================================
        # Train
        # ==============================================
        t0_train = time.time()
        model.train()

        epoch_loss, n_steps, tre_before, tre_after = 0, 0, 0, 0
        for batch_data in tqdm(train_loader):
            # Get data
            fixed_image = batch_data["fixed_image"].to(device)
            moving_image = batch_data["moving_image"].to(device)
            moving_label = batch_data["moving_label"].to(device)
            fixed_label = batch_data["fixed_label"].to(device)
            fixed_keypoints = batch_data["fixed_keypoints"].to(device)
            moving_keypoints = batch_data["moving_keypoints"].to(device)
            n_steps += 1
            # Forward pass and loss
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                    fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
                )
                loss = loss_fun(
                    fixed_image,
                    pred_image,
                    fixed_label,
                    pred_label,
                    fixed_keypoints + ddf_keypoints,
                    moving_keypoints,
                    ddf_image,
                    lam_t,
                    lam_l,
                    lam_m,
                    lam_r,
                )
            # Optimise
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # TRE before (voxel space)
            tre_before += tre(fixed_keypoints, moving_keypoints)
            tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints)

        # Scheduler step
        lr_scheduler.step()
        # Loss
        epoch_loss /= n_steps
        log_train_loss.append(epoch_loss)
        if do_save:
            writer.add_scalar("train_loss", epoch_loss, epoch)
        print(f"{epoch + 1} | loss={epoch_loss:.6f}")

        # Mean TRE
        tre_before /= n_steps
        tre_after /= n_steps
        print(
            (
                f"{epoch + 1} | tre_before_train={tre_before:.3f}, tre_after_train={tre_after:.3f}, "
                f"elapsed time: {time.time()-t0_train:.2f} sec."
            )
        )

        # ==============================================
        # Eval
        # ==============================================
        if (epoch + 1) % val_interval == 0:
            t0_eval = time.time()
            model.eval()

            n_steps, tre_before, tre_after = 0, 0, 0
            with torch.no_grad():
                for batch_data in tqdm(val_loader):
                    # Get data
                    fixed_image = batch_data["fixed_image"].to(device)
                    moving_image = batch_data["moving_image"].to(device)
                    moving_label = batch_data["moving_label"].to(device)
                    fixed_label = batch_data["fixed_label"].to(device)
                    fixed_keypoints = batch_data["fixed_keypoints"].to(device)
                    moving_keypoints = batch_data["moving_keypoints"].to(device)
                    n_steps += 1
                    # Infer
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                            fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
                        )
                    # TRE
                    tre_before += tre(fixed_keypoints, moving_keypoints, vx=vx)
                    tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints, vx=vx)
                    # Dice
                    pred_label = pred_label.round()
                    dice_metric_before(y_pred=moving_label, y=fixed_label)
                    dice_metric_after(y_pred=pred_label, y=fixed_label)

            # Dice
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            if do_save:
                writer.add_scalar("val_dice", dice_after, epoch)
            log_val_dice.append(dice_after)
            print(f"{epoch + 1} | dice_before ={dice_before:.3f}, dice_after ={dice_after:.3f}")

            # Mean TRE
            tre_before /= n_steps
            tre_after /= n_steps
            log_val_tre.append(tre_after.item())
            if do_save:
                writer.add_scalar("val_tre", tre_after, epoch)
            print(
                (
                    f"{epoch + 1} | tre_before_val ={tre_before:.3f}, tre_after_val ={tre_after:.3f}, "
                    f"elapsed time: {time.time()-t0_train:.2f} sec."
                )
            )

            if tre_after < best_eval_tre:
                best_eval_tre = tre_after
                if do_save:
                    # Save best model based on TRE
                    if pth_best_tre != "":
                        os.remove(os.path.join(dir_save, pth_best_tre))
                    pth_best_tre = f"segresnet_kpt_loss_best_tre_{epoch + 1}_{best_eval_tre:.3f}.pth"
                    torch.save(model.state_dict(), os.path.join(dir_save, pth_best_tre))
                    print(f"{epoch + 1} | Saving best TRE model: {pth_best_tre}")

            if dice_after > best_eval_dice:
                best_eval_dice = dice_after
                if do_save:
                    # Save best model based on Dice
                    if pth_best_dice != "":
                        os.remove(os.path.join(dir_save, pth_best_dice))
                    pth_best_dice = f"segresnet_kpt_loss_best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                    torch.save(model.state_dict(), os.path.join(dir_save, pth_best_dice))
                    print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")

        if do_save:
            # Save latest model
            if pth_latest != "":
                os.remove(os.path.join(dir_save, pth_latest))
            pth_latest = "segresnet_kpt_loss_latest.pth"
            torch.save(model.state_dict(), os.path.join(dir_save, pth_latest))

    # log_val_tre = [x.item() for x in log_val_tre]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(log_train_loss)
    axs[0].title.set_text("train_loss")
    axs[1].plot(log_val_dice)
    axs[1].title.set_text("val_dice")
    axs[2].plot(log_val_tre)
    axs[2].title.set_text("val_tre")
    plt.show()

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
        filename_best_model = glob.glob(os.path.join(dir_load, "segresnet_kpt_loss_best_tre*"))[0]
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
        with torch.autocast("cuda", enabled=amp_enabled):
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
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2], s=2.0, marker="o", color="lightblue")
    ax.scatter(moving_points[:, 0], moving_points[:, 1], moving_points[:, 2], s=2.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    # After registration
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(moved_keypoints[:, 0], moved_keypoints[:, 1], moved_keypoints[:, 2], s=2.0, marker="o", color="lightblue")
    ax.scatter(moving_keypoints[:, 0], moving_keypoints[:, 1], moving_keypoints[:, 2], s=2.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.show()
