import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    ApplyTransformToPointsd,
    SqueezeDimd,
)
from monai.transforms import (
    MapTransform,
)
from monai.utils import ensure_tuple_rep, ensure_tuple


class LoadKeypointsd(MapTransform):
    """
    Load keypoints from csv file
    """

    def __init__(self, keys, allow_missing_keys=True, refer_keys=None):
        super().__init__(keys, allow_missing_keys)
        self.refer_keys = ensure_tuple_rep(None, len(self.keys)) if refer_keys is None else ensure_tuple(refer_keys)

    def __call__(self, data):
        d = dict(data)
        for key, refer_key in self.key_iterator(d, self.refer_keys):
            keypoints = d[key]
            keypoints = pd.read_csv(keypoints, header=None)
            keypoints = keypoints.to_numpy()
            if refer_key is not None:
                # assume keypoints already have affine applied
                keypoints = MetaTensor(keypoints, affine=d[refer_key].affine)
            else:
                keypoints = MetaTensor(keypoints)
            d[key] = keypoints  # [N, 3]
        return d


def get_train_transforms(spatial_size, target_res):
    train_transforms = Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            LoadKeypointsd(keys=["fixed_keypoints", "moving_keypoints"], refer_keys=["fixed_image", "moving_image"]),
            EnsureChannelFirstd(keys=["fixed_keypoints", "moving_keypoints"], channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-1200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=spatial_size,
            ),
            RandAffined(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                prob=0.8,
                shear_range=0.2,
                translate_range=int(25 * target_res[0] / 224),
                rotate_range=np.pi / 180 * 15,
                scale_range=0.2,
                padding_mode=("zeros", "zeros", "zeros", "zeros"),
            ),
            ApplyTransformToPointsd(
                keys=["fixed_keypoints", "moving_keypoints"],
                refer_keys=["fixed_image", "moving_image"],
                dtype=torch.float32,
            ),
            SqueezeDimd(keys=["fixed_keypoints", "moving_keypoints"]),  # remove channel dim
        ]
    )
    return train_transforms


def get_val_transforms(spatial_size):
    val_transforms = Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            LoadKeypointsd(keys=["fixed_keypoints", "moving_keypoints"], refer_keys=["fixed_image", "moving_image"]),
            EnsureChannelFirstd(keys=["fixed_keypoints", "moving_keypoints"], channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-1200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=spatial_size,
            ),
            ApplyTransformToPointsd(
                keys=["fixed_keypoints", "moving_keypoints"],
                refer_keys=["fixed_image", "moving_image"],
                dtype=torch.float32,
            ),
            SqueezeDimd(keys=["fixed_keypoints", "moving_keypoints"]),  # remove channel dim
        ]
    )
    return val_transforms
