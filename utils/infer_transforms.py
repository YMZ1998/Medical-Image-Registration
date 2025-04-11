from monai.transforms import Compose, LoadImage, Resize, ScaleIntensityRange, ToTensor, EnsureType
import torch


def get_transforms(spatial_size, normalize=True):
    transforms = [
        LoadImage(image_only=True, ensure_channel_first=True),
        Resize(spatial_size, mode="trilinear", align_corners=True),
    ]
    if normalize:
        transforms.insert(1, ScaleIntensityRange(a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True))
    transforms += [ToTensor(), EnsureType(dtype=torch.float32)]
    return Compose(transforms)


def load_image(image_path, spatial_size, normalize=True):
    return get_transforms(spatial_size, normalize)(image_path).unsqueeze(0)
