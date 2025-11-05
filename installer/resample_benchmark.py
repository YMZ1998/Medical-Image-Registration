import monai
from monai.apps.utils import download_url
from monai.config import print_config

import itk
import SimpleITK as sitk  # noqa: N813

import torch
import os
import shutil
import tempfile
import PIL
import numpy as np

from matplotlib import pyplot as plt

directory = "./test_data"
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"root dir is: {root_dir}")

# dict of file name and corresponding urls
url_dict = {
    "monai.png": "https://github.com/Project-MONAI/project-monai.github.io/raw/master/assets/logo/MONAI-logo_color.png",
    # noqa: E501
    "mri.nii": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/Prostate_T2W_AX_1.nii",
}


# # download and rename
# for k, v in url_dict.items():
#     download_url(v, f"{root_dir}/{k}")

def transformation_display(img_path, tx, ty, theta):
    """
    transform a image by MONAI and SimpleITK and plot both transformed images
    """
    # monai transform
    monai_img = np.asarray(PIL.Image.open(img_path))
    monai_img = monai_img.transpose((2, 0, 1))  # CHW
    monai_img = torch.tensor(monai_img, dtype=torch.float32)

    affine = monai.transforms.Affine(
        rotate_params=theta,  # radians
        translate_params=(tx, ty),  # pixels
        padding_mode="zeros"
    )

    monai_transformed = affine(monai_img)[0].permute((1, 2, 0))  # back to HWC

    # sitk transform
    sitk_img = sitk.ReadImage(img_path)
    euler2d_transform = sitk.Euler2DTransform()
    euler2d_transform.SetCenter(
        sitk_img.TransformContinuousIndexToPhysicalPoint(np.array(sitk_img.GetSize()) / 2.0)
    )
    euler2d_transform.SetTranslation((tx, ty))
    euler2d_transform.SetAngle(theta)
    sitk_transformed = sitk.Resample(sitk_img, euler2d_transform)

    # plot monai and sitk transformed results
    fig, ax = plt.subplots(2)
    ax[0].imshow(monai_transformed.numpy().astype(np.uint8))
    ax[0].axis("off")
    ax[0].set_title("MONAI")
    ax[1].imshow(sitk.GetArrayFromImage(sitk_transformed))
    ax[1].axis("off")
    ax[1].set_title("SimpleITK")
    plt.show()


img_path = f"{root_dir}/test.png"
# 示例调用（固定变换，不是范围）
# transformation_display(img_path, tx=20, ty=10, theta=np.pi / 12)
# Read image
monai_img = np.asarray(PIL.Image.open(img_path))
monai_img = monai_img.transpose((2, 0, 1))
monai_img = torch.tensor(monai_img, dtype=torch.float)
sitk_img = sitk.ReadImage(img_path)

width, height = sitk_img.GetSize()

# Generate random samples inside the image, we will obtain the intensity/color values at these points.
num_samples = 10
physical_points = np.array([np.random.randint(monai_img.shape[1:]) for _ in range(10)], dtype=float)  # (10, 2)

# initialise ddf as a zero matrix
ddf = torch.zeros(2, height, width).to(torch.float)  # (2, H, W)
# add displacement of y coordinate to sampled locations
for i, pnt in enumerate(physical_points.astype(int)):
    ddf[0, pnt[0], pnt[1]] += i

# ddf = (torch.rand(2, height, width) - 0.5) * 10.0  # [-5, 5]
print("ddf :", ddf )

# initialise warp layer
warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")

# warp, note an batch dimension is added and removed during calculation
monai_resample = warp(monai_img.unsqueeze(0), ddf.unsqueeze(0).to(monai_img)).squeeze(0)  # (3, H, W)

# Create an image of size [width, height]. The pixel type is irrelevant, as the image is
# just defining the interpolation grid (sitkUInt8 has minimal memory footprint).
interp_grid_img = sitk.Image([width, height], sitk.sitkUInt8)

# initialise displacement
sitk_displacement_img = sitk.Image([width, height], sitk.sitkVectorFloat64, sitk_img.GetDimension())
# add displacement of y coordinate to sampled locations
for i, pnt in enumerate(physical_points):
    sitk_displacement_img[int(pnt[1]), int(pnt[0])] = np.array([0, i], dtype=float)

# select linear interpolater to match `bilinear` mode in MONAI
interpolator_enum = sitk.sitkLinear
# set default_output_pixel_value to 0.0 to match `zero` padding_mode in MONAI
default_output_pixel_value = 0.0
# set output_pixel_type
output_pixel_type = sitk.sitkVectorFloat32
# resample
sitk_resample = sitk.Resample(
    sitk_img,
    interp_grid_img,
    sitk.DisplacementFieldTransform(sitk_displacement_img),
    interpolator_enum,
    default_output_pixel_value,
    output_pixel_type,
)

# turn resampled image into numpy array for later comparison
sitk_resample = sitk.GetArrayFromImage(sitk_resample)

for _, pnt in enumerate(physical_points.astype(int)):
    print(
        f"at location {pnt}: original intensity {monai_img[:, pnt[0], pnt[1]]} "
        + f"resampled to {monai_resample[:, pnt[0], pnt[1]]} by MONAI and {sitk_resample[pnt[0], pnt[1]]} by SITK"
    )

