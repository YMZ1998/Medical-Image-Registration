import os
import time
import torch
from monai.metrics import DiceMetric
from tqdm import tqdm

from utils.utils import tre, forward


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_fun, warp_layer, device, args, writer=None):
    """
    Train the model for one epoch.

    Parameters:
    - model: The neural network model.
    - train_loader: DataLoader for training data.
    - optimizer: Optimizer for training.
    - lr_scheduler: Learning rate scheduler.
    - loss_fun: Loss function.
    - warp_layer: Warping layer for transformation.
    - device: Device to run training on (e.g., 'cuda' or 'cpu').
    - args: Arguments containing training configurations (e.g., AMP usage, tensorboard flag).
    - writer: TensorBoard writer (optional).

    Returns:
    - epoch_loss: Average loss for the epoch.
    """
    # loss weights (set to zero to disable loss term)
    lam_t = 1e0  # TRE  (keypoint loss)
    lam_l = 0  # Dice (mask overlay)
    lam_m = 0  # MSE (image similarity)
    lam_r = 0  # Bending loss (smoothness of the DDF)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    t0_train = time.time()
    model.train()

    epoch_loss, n_steps, tre_before, tre_after = 0, 0, 0, 0
    for batch_data in tqdm(train_loader, desc="Training Epoch"):
        fixed_image = batch_data["fixed_image"].to(device)
        moving_image = batch_data["moving_image"].to(device)
        moving_label = batch_data["moving_label"].to(device)
        fixed_label = batch_data["fixed_label"].to(device)
        fixed_keypoints = batch_data["fixed_keypoints"].to(device)
        moving_keypoints = batch_data["moving_keypoints"].to(device)
        n_steps += 1

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
            )
            loss = loss_fun(
                fixed_image, pred_image, fixed_label, pred_label,
                fixed_keypoints + ddf_keypoints, moving_keypoints, ddf_image,
                lam_t, lam_l, lam_m, lam_r
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        tre_before += tre(fixed_keypoints, moving_keypoints)
        tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints)

    lr_scheduler.step()
    epoch_loss /= n_steps

    if writer and args.tensorboard:
        writer.add_scalar("train_loss", epoch_loss)

    print(f"Loss={epoch_loss:.6f}")
    print(
        f"TRE Before={tre_before / n_steps:.3f}, TRE After={tre_after / n_steps:.3f}, "
        f"Elapsed time: {time.time() - t0_train:.2f} sec."
    )

    return epoch_loss


def evaluate_model(model, warp_layer, val_loader, device, args, vx, writer=None):
    """
    Evaluate the model on validation data.
    """
    t0_eval = time.time()
    model.eval()

    n_steps, tre_before, tre_after = 0, 0, 0
    dice_metric_before, dice_metric_after = DiceMetric(), DiceMetric()

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation Epoch"):
            fixed_image = batch_data["fixed_image"].to(device)
            moving_image = batch_data["moving_image"].to(device)
            moving_label = batch_data["moving_label"].to(device)
            fixed_label = batch_data["fixed_label"].to(device)
            fixed_keypoints = batch_data["fixed_keypoints"].to(device)
            moving_keypoints = batch_data["moving_keypoints"].to(device)
            n_steps += 1

            with torch.cuda.amp.autocast(enabled=args.amp):
                ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                    fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
                )

            tre_before += tre(fixed_keypoints, moving_keypoints, vx=vx)
            tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints, vx=vx)

            pred_label = pred_label.round()
            dice_metric_before(y_pred=moving_label, y=fixed_label)
            dice_metric_after(y_pred=pred_label, y=fixed_label)

    dice_before = dice_metric_before.aggregate().item()
    dice_metric_before.reset()
    dice_after = dice_metric_after.aggregate().item()
    dice_metric_after.reset()

    if writer and args.tensorboard:
        writer.add_scalar("val_dice", dice_after)

    print(f"Dice Before={dice_before:.3f}, Dice After={dice_after:.3f}")

    tre_before /= n_steps
    tre_after /= n_steps

    if writer and args.tensorboard:
        writer.add_scalar("val_tre", tre_after)

    print(
        f"TRE Before={tre_before:.3f}, TRE After={tre_after:.3f}, "
        f"Elapsed time: {time.time() - t0_eval:.2f} sec."
    )

    return tre_after, dice_after


def save_best_model(model, epoch, metric, best_metric, path_prefix, suffix, dir_save, prev_path):
    if (suffix == "tre" and metric < best_metric) or (suffix == "dice" and metric > best_metric):
        if prev_path != "":
            os.remove(os.path.join(dir_save, prev_path))
        filename = f"{path_prefix}_kpt_loss_best_{suffix}_{epoch + 1}_{metric:.3f}.pth"
        torch.save(model.state_dict(), os.path.join(dir_save, filename))
        print(f"{epoch + 1} | Saving best {suffix.upper()} model: {filename}")
        return filename, metric
    return prev_path, best_metric


def save_latest_model(model, path_prefix, dir_save, prev_path):
    if prev_path != "":
        os.remove(os.path.join(dir_save, prev_path))
    filename = f"{path_prefix}_kpt_loss_latest.pth"
    torch.save(model.state_dict(), os.path.join(dir_save, filename))
    return filename
