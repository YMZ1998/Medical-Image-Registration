import torch
import monai
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

# 读取图像
img_path = "./test_data/test.png"

# 读取图像
img = Image.open(img_path)  # PIL Image 对象

# # 缩小一半
# scale_factor = 1
# new_size = (int(img.width * scale_factor), int(img.height * scale_factor))  # PIL Image 用 width/height
# img_down = img.resize(new_size, Image.BILINEAR)

# 转回 numpy array
monai_img = np.asarray(img)
print("monai_img.shape:", monai_img.shape)
height, width = monai_img.shape[:2]
monai_img_torch = torch.tensor(monai_img.transpose(2, 0, 1), dtype=torch.float32)  # (C,H,W)

amplitude = 20.0  # 位移幅度
# 生成网格
y = torch.linspace(0, 2*np.pi, height)  # 行
x = torch.linspace(0, 2*np.pi, width)   # 列
Y, X = torch.meshgrid(y, x, indexing='ij')  # (H, W)

# 构造 DDF
ddf = torch.zeros(2, height, width)
ddf[0] = amplitude * torch.sin(Y * 2)  # x方向弯曲
ddf[1] = amplitude * torch.sin(X * 2)  # y方向弯曲
print("ddf:", ddf)

# ---- MONAI warp ----
warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
monai_resample = warp(monai_img_torch.unsqueeze(0), ddf.unsqueeze(0).to(monai_img_torch)).squeeze(0)  # (3,H,W)
monai_resample_np = monai_resample.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# ---- SITK warp ----
sitk_img = sitk.ReadImage(img_path)
interp_grid_img = sitk.Image([width, height], sitk.sitkUInt8)

sitk_displacement_img = sitk.Image([width, height], sitk.sitkVectorFloat64, 2)
ddf_np = ddf.permute(1, 2, 0).numpy()  # (H,W,2)
for y in range(height):
    for x in range(width):
        sitk_displacement_img.SetPixel(x, y, [float(ddf_np[y, x, 0]), float(ddf_np[y, x, 1])])

ddf_sitk_img = sitk.GetImageFromArray(ddf_np, isVector=True)
sitk.WriteImage(ddf_sitk_img, "./test_data/ddf_field.nii.gz")

sitk_resample = sitk.Resample(
    sitk_img,
    interp_grid_img,
    sitk.DisplacementFieldTransform(sitk_displacement_img),
    sitk.sitkLinear,
    0.0,
    sitk.sitkVectorFloat32,
)
sitk_resample_np = sitk.GetArrayFromImage(sitk_resample)

# ---- 可选的采样点（物理点） ----
num_samples = 10
physical_points = np.zeros((num_samples, 2))
physical_points[:, 0] = np.random.randint(0, width, size=num_samples)   # x
physical_points[:, 1] = np.random.randint(0, height, size=num_samples)  # y

# ---- 可视化 ----
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1️⃣ 原始图像
axs[0, 0].imshow(monai_img)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

# 2️⃣ MONAI 变形结果
axs[0, 1].imshow(monai_resample_np)
axs[0, 1].set_title("MONAI warped image")
axs[0, 1].axis("off")

# 3️⃣ SimpleITK 变形结果
axs[1, 0].imshow(sitk_resample_np)
axs[1, 0].set_title("SimpleITK warped image")
axs[1, 0].axis("off")

# 4️⃣ DDF 可视化（矢量场）
skip = 20  # 每隔多少像素显示一个箭头
Y, X = np.mgrid[0:height:skip, 0:width:skip]
U = ddf[0, ::skip, ::skip].cpu().numpy()
V = -ddf[1, ::skip, ::skip].cpu().numpy()  # y轴反向显示更直观
axs[1, 1].imshow(monai_img)
axs[1, 1].quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1)
axs[1, 1].set_title("Random DDF field (quiver)")
axs[1, 1].axis("off")

# 可选：绘制采样点
for ax in axs.ravel():
    ax.plot(physical_points[:, 0], physical_points[:, 1], 'bo', markersize=5)

plt.tight_layout()
plt.show()

for pnt in physical_points:
    x, y = pnt.astype(int)
    # 防止溢出
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    orig_intensity = monai_img[y, x, :]
    monai_val = monai_resample_np[y, x, :]
    sitk_val = sitk_resample_np[y, x, :]

    print(f"at location (x={x}, y={y}): "
          f"original={orig_intensity}, "
          f"MONAI={monai_val}, "
          f"SITK={sitk_val}")
