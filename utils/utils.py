import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.data.utils import list_data_collate
from monai.losses import BendingEnergyLoss, DiceLoss
from torch.nn import MSELoss


def forward(fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer):
    """
    Model forward pass: predict DDF, warp moving images/labels/keypoints
    """
    batch_size = fixed_image.shape[0]

    ddf_image = model(torch.cat((moving_image, fixed_image), dim=1)).float()

    pred_image = warp_layer(moving_image, ddf_image)

    if moving_label is not None:
        pred_label = warp_layer(moving_label, ddf_image)
    else:
        pred_label = None

    if fixed_keypoints is not None:
        with torch.no_grad():
            offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
            offset = offset[None][None]
            ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
        ddf_keypoints = (
            F.grid_sample(ddf_image, ddf_keypoints.view(batch_size, -1, 1, 1, 3))
            .view(batch_size, 3, -1)
            .permute((0, 2, 1))
        )
    else:
        ddf_keypoints = None

    return ddf_image, ddf_keypoints, pred_image, pred_label


def collate_fn(batch):
    """
    Custom collate function.
    Some background:
        Collation is the "collapsing" of a list of N-dimensional tensors into a single (N+1)-dimensional tensor.
        The `Dataloader` object  performs this step after receiving a batch of (transformed) data from the
        `Dataset` object.
        Note that the `Resized` transform above resamples all image tensors to a shape `spatial_size`,
        thus images can be easily collated.
        Keypoints, however, are of different row-size and thus cannot be easily collated
        (a.k.a. "ragged" or "jagged" tensors): [(n_0, 3), (n_1, 3), ...]
        This function aligns the row-size of these tensors such that they can be collated like
        any regular list of tensors.
        To do this, the max number of keypoints is determined, and shorter keypoint-lists are filled up with NaNs.
        Then, the average-TRE loss below can be computed via `nanmean` aggregation (i.e. ignoring filled-up elements).
    """
    max_length = 0
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        if length > max_length:
            max_length = length
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        data["fixed_keypoints"] = torch.concat(
            (data["fixed_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )
        data["moving_keypoints"] = torch.concat(
            (data["moving_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )

    return list_data_collate(batch)


def tre(fixed, moving, vx=None):
    if vx is None:
        return ((fixed - moving) ** 2).sum(-1).sqrt().nanmean()
    else:
        return ((fixed - moving).mul(vx) ** 2).sum(-1).sqrt().nanmean()


def loss_fun(
    fixed_image,
    pred_image,
    fixed_label,
    pred_label,
    fixed_keypoints,
    pred_keypoints,
    ddf_image,
    lam_t=1.0,
    lam_l=0.0,
    lam_m=0.0,
    lam_r=0.0,
):
    """
    Multi-target loss function:
        - TRE (Target Registration Error) as main loss
        - Optionally includes MSE, Dice loss, and Bending Energy regularization
    """
    loss_terms = []

    if lam_t > 0:
        loss_terms.append(lam_t * tre(fixed_keypoints, pred_keypoints))
    if lam_l > 0:
        loss_terms.append(lam_l * DiceLoss()(pred_label, fixed_label))
    if lam_m > 0:
        loss_terms.append(lam_m * MSELoss()(fixed_image, pred_image))
    if lam_r > 0:
        loss_terms.append(lam_r * BendingEnergyLoss()(ddf_image))

    return sum(loss_terms) if loss_terms else 0.0


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_best_model(model, model_dir):
    best_model_files = glob.glob(os.path.join(model_dir, "*_kpt_loss_best_tre*"))
    if not best_model_files:
        raise FileNotFoundError("No best model checkpoint found.")
    checkpoint_path = best_model_files[0]
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    return model


def to_numpy_list(tensor_list):
    return [t.detach().cpu().item() if hasattr(t, 'detach') else t for t in tensor_list]


def plot_training_logs(logs, titles, save_path=None, figsize=(15, 5)):
    num_plots = len(logs)
    fig, axs = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axs = [axs]

    for i in range(num_plots):
        logs[i] = to_numpy_list(logs[i])

        axs[i].plot(logs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Value")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    log_train_loss = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    log_val_dice = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32)
    log_val_tre = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    plot_training_logs(
        [log_train_loss, log_val_dice, log_val_tre],
        ["Train Loss", "Validation Dice", "Validation TRE"],
        # "training_logs.png"
    )
