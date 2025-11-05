import itk
import monai
import numpy as np
import torch
import SimpleITK as sitk
from matplotlib import pyplot as plt

# 读取图像
root_dir = "./test_data"
img = monai.transforms.LoadImaged(keys="img")({"img": f"{root_dir}/mri.nii"})["img"]
# W, H, D -> D, H, W
img = img.permute((2, 1, 0))

# 随机位移场
ddf = np.random.random((3, *img.shape)).astype(np.float32)  # (3, D, H, W)
ddf[0] = ddf[0] * img.shape[0] * 0.1
ddf[1] = ddf[1] * img.shape[1] * 0.1
ddf[2] = ddf[2] * img.shape[2] * 0.1

# Initialise MONAI warp layer
warp_layer = monai.networks.blocks.Warp(padding_mode="zeros")

# Convert image and ddf to tensor, add channel dim to image
monai_img = torch.tensor(img).unsqueeze(0)
monai_ddf = torch.tensor(ddf)

# Warp the image using MONAI
monai_warped_img = warp_layer(monai_img.unsqueeze(0), monai_ddf.unsqueeze(0)).squeeze(0)

# Remove channel dim and convert to numpy
monai_warped_img = np.asarray(monai_warped_img.squeeze(0))

# ---- 使用 SimpleITK 进行变形 ----
# Convert to SimpleITK format
sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
sitk_img.SetSpacing((1.0, 1.0, 1.0))  # Set spacing to 1 (adjust as necessary)

# Create displacement field for SimpleITK
# Correctly transpose ddf to (D, H, W, 3) for displacement field
sitk_ddf = sitk.GetImageFromArray(ddf.transpose(1, 2, 3, 0).astype(np.float64))  # (D, H, W, 3)
sitk_ddf.SetSpacing(sitk_img.GetSpacing())  # Keep the spacing same as original image

# Set up SimpleITK warp filter
warp_filter = sitk.ResampleImageFilter()
warp_filter.SetSize(sitk_img.GetSize())
warp_filter.SetOutputSpacing(sitk_img.GetSpacing())
warp_filter.SetOutputOrigin(sitk_img.GetOrigin())
warp_filter.SetOutputDirection(sitk_img.GetDirection())
warp_filter.SetTransform(sitk.DisplacementFieldTransform(sitk_ddf))  # Ensure displacement field is of type sitkVectorFloat64
warp_filter.SetDefaultPixelValue(0)
warp_filter.SetInterpolator(sitk.sitkLinear)

# Perform the warp using SimpleITK
sitk_warped_img = warp_filter.Execute(sitk_img)

# Convert SimpleITK image to numpy
itk_warped_img = sitk.GetArrayFromImage(sitk_warped_img)

# Visualization
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
        row[2].set_title("SimpleITK")
plt.show()

# Compute mean absolute difference
diff = np.mean(
    np.divide(
        itk_warped_img - monai_warped_img, itk_warped_img, out=np.zeros_like(itk_warped_img), where=itk_warped_img != 0
    )
)
print(f"MONAI warped image mean absolute difference: {diff}")
