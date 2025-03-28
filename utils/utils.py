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

    # predict DDF through LocalNet
    ddf_image = model(torch.cat((moving_image, fixed_image), dim=1)).float()

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf_image)

    # warp moving label (optional)
    if moving_label is not None:
        pred_label = warp_layer(moving_label, ddf_image)
    else:
        pred_label = None

    # warp vectors for keypoints (optional)
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
    """
    Computes target registration error (TRE) loss for keypoint matching.
    """
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
    lam_t,
    lam_l,
    lam_m,
    lam_r,
):
    """
    Custom multi-target loss:
        - TRE as main loss component
        - Parametrizable weights for further (optional) components: MSE/BendingEnergy/Dice loss
    Note: Might require "calibration" of lambda weights for the multi-target components,
        e.g. by making a first trial run, and manually setting weights to account for different magnitudes
    """
    # Instantiate where necessary
    if lam_m > 0:
        mse_loss = MSELoss()
    if lam_r > 0:
        regularization = BendingEnergyLoss()
    if lam_l > 0:
        label_loss = DiceLoss()
    # Compute loss components
    t = tre(fixed_keypoints, pred_keypoints) if lam_t > 0 else 0.0
    p = label_loss(pred_label, fixed_label) if lam_l > 0 else 0.0
    m = mse_loss(fixed_image, pred_image) if lam_m > 0 else 0.0
    r = regularization(ddf_image) if lam_r > 0 else 0.0
    # Weighted combination:
    return lam_t * t + lam_l * p + lam_m * m + lam_r * r