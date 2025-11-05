import itk
import monai
import numpy as np
import torch

root_dir = "./test_data"
from matplotlib import pyplot as plt

img = monai.transforms.LoadImaged(keys="img")({"img": f"{root_dir}/mri.nii"})["img"]
# W, H, D -> D, H, W
img = img.permute((2, 1, 0))

ddf = np.random.random((3, *img.shape)).astype(np.float32)  # (3, D, H, W)
ddf[0] = ddf[0] * img.shape[0] * 0.1
ddf[1] = ddf[1] * img.shape[1] * 0.1
ddf[2] = ddf[2] * img.shape[2] * 0.1

# initialise warp layer
warp_layer = monai.networks.blocks.Warp(padding_mode="zeros")

# turn image and ddf to tensor, add channel dim to image
monai_img = torch.tensor(img).unsqueeze(0)
monai_ddf = torch.tensor(ddf)

# warp, note an batch dimension is added and removed during calculation
monai_warped_img = warp_layer(monai_img.unsqueeze(0), monai_ddf.unsqueeze(0)).squeeze(0)

# remove channel dim and transform to numpy array
monai_warped_img = np.asarray(monai_warped_img.squeeze(0))

Dimension = 3

# initialise image
PixelType = itk.F  # similar to np.float32
ImageType = itk.Image[PixelType, Dimension]
# cast image to ImageType
itk_img = itk.PyBuffer[ImageType].GetImageFromArray(img.astype(np.float32), is_vector=None)

# initialise displacement
VectorComponentType = itk.F  # similar to np.float32
VectorPixelType = itk.Vector[VectorComponentType, Dimension]
DisplacementFieldType = itk.Image[VectorPixelType, Dimension]
# 3, D, H, W -> D, H, W, 3
itk_ddf = ddf.transpose((1, 2, 3, 0))
# x, y, z -> z, x, y
itk_ddf = itk_ddf[..., ::-1]
# cast ddf to DisplacementFieldType
deformation_field = itk.PyBuffer[DisplacementFieldType].GetImageFromArray(itk_ddf.astype(np.float32), is_vector=True)
# initialise warpFilter, set input, output and displacement field types
warp_filter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()
# set interpolator
interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
warp_filter.SetInterpolator(interpolator)
# set output spacing, origin and direction
warp_filter.SetOutputSpacing(itk_img.GetSpacing())
warp_filter.SetOutputOrigin(itk_img.GetOrigin())
warp_filter.SetOutputDirection(itk_img.GetDirection())
# warp
warp_filter.SetDisplacementField(deformation_field)
warp_filter.SetInput(itk_img)
warp_filter.Update()
itk_warped_img = warp_filter.GetOutput()

# transform itk.Image to numpy array
itk_warped_img = np.asarray(itk_warped_img)
fig, ax = plt.subplots(3, 3)
for i, (row, d) in enumerate(zip(ax, [5, 10, 15])):
    row[0].imshow(img[d], cmap="gray")
    row[0].axis("off")
    row[1].imshow(monai_warped_img[d], cmap="gray")
    row[1].axis("off")
    row[2].imshow(itk_warped_img[d], cmap="gray")
    row[2].axis("off")

    if i == 0:
        row[0].set_title("original img")
        row[1].set_title("MONAI")
        row[2].set_title("python itk")
plt.show()

diff = np.mean(
    np.divide(
        itk_warped_img - monai_warped_img, itk_warped_img, out=np.zeros_like(itk_warped_img), where=itk_warped_img != 0
    )
)
print(f"MONAI warped image mean absolute difference: {diff}")
