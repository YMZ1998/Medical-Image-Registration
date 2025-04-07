import os
import warnings
from pprint import pprint

import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.utils import set_determinism, first
from torch.utils.tensorboard import SummaryWriter

from parse_args import get_device, parse_args, get_net
from utils.dataset import get_files
from utils.train_and_eval import train_one_epoch, evaluate_model, save_best_model, save_latest_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.utils import forward, loss_fun, collate_fn, plot_training_logs
from val import visualize_registration

if __name__ == "__main__":
    # === Setup ===
    set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    args = parse_args()
    device = get_device()
    train_files, val_files = get_files(os.path.join(args.data_path, "NLST"))
    pprint(train_files[:2])

    # === Resolution Config ===
    full_res_training = False
    target_res = [224, 192, 224] if full_res_training else [96, 96, 96]
    spatial_size = [-1, -1, -1] if full_res_training else target_res
    vx = torch.tensor(np.array([1.5, 1.5, 1.5]) / (np.array(target_res) / np.array([224, 192, 224]))).to(device)

    # === Logging and Saving Paths ===
    print(f'tensorboard --logdir="./models/nlst/{args.arch}"')
    writer = SummaryWriter(log_dir=args.model_dir) if args.tensorboard else None

    # === Datasets and Loaders ===
    train_ds = Dataset(train_files, transform=get_train_transforms(spatial_size, target_res))
    val_ds = Dataset(val_files, transform=get_val_transforms(spatial_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # === Model, Optimizer, Scheduler ===
    model = get_net(args).to(device)
    warp_layer = Warp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Metrics ===
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # === Training Loop ===
    log_train_loss, log_val_dice, log_val_tre = [], [], []
    best_eval_tre, best_eval_dice = float("inf"), 0.0
    pth_best_tre, pth_best_dice, pth_latest = "", "", ""

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_fun, warp_layer, device, args,
                                     writer)
        log_train_loss.append(epoch_loss)

        tre_after, dice_after = evaluate_model(model, warp_layer, val_loader, device, args, vx, writer)
        log_val_tre.append(tre_after)
        log_val_dice.append(dice_after)

        pth_best_tre, best_eval_tre = save_best_model(model, epoch, tre_after, best_eval_tre, args.arch, "tre",
                                                      args.model_dir, pth_best_tre)
        pth_best_dice, best_eval_dice = save_best_model(model, epoch, dice_after, best_eval_dice, args.arch, "dice",
                                                        args.model_dir, pth_best_dice)
        pth_latest = save_latest_model(model, args.arch, args.model_dir, pth_latest)

    plot_training_logs(
        [log_train_loss, log_val_dice, log_val_tre],
        ["Train Loss", "Validation Dice", "Validation TRE"],
        os.path.join(args.model_dir, "training_logs.png")
    )

    # === Evaluation and Visualization ===
    set_determinism(seed=1)
    check_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)

    # Inference
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

    # Visualization
    visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res)

