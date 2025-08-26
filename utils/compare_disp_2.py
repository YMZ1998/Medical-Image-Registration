import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import map_coordinates


# -----------------------------
# compose_disp 函数
def compose_disp(u1, u2):
    Z, Y, X, _ = u1.shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    phi_z = zz + u1[..., 0]
    phi_y = yy + u1[..., 1]
    phi_x = xx + u1[..., 2]
    coords = [phi_z, phi_y, phi_x]
    u2z = map_coordinates(u2[..., 0], coords, order=3, mode='nearest')
    u2y = map_coordinates(u2[..., 1], coords, order=3, mode='nearest')
    u2x = map_coordinates(u2[..., 2], coords, order=3, mode='nearest')
    comp = np.stack([u1[..., 0] + u2z,
                     u1[..., 1] + u2y,
                     u1[..., 2] + u2x], axis=-1)
    return comp


# -----------------------------
# 1) 生成模拟位移场
Z, Y, X = 32, 32, 32
u_AB = np.zeros((Z, Y, X, 3), dtype=np.float32)
u_BA = np.zeros((Z, Y, X, 3), dtype=np.float32)

zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
# 平滑小位移
dx=0.5
u_AB[..., 0] = dx * np.sin(2 * np.pi * yy / Y)
u_AB[..., 1] = dx * np.sin(2 * np.pi * xx / X)
u_AB[..., 2] = dx * np.sin(2 * np.pi * zz / Z)
u_BA = -u_AB  # 反向

# 保存为 NIfTI
sitk.WriteImage(sitk.GetImageFromArray(u_AB, isVector=True), "sim_u_AB.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(u_BA, isVector=True), "sim_u_BA.nii.gz")

# -----------------------------
# 3) 组合位移
comp = compose_disp(u_AB, u_BA)
sitk.WriteImage(sitk.GetImageFromArray(comp, isVector=True), "sim_comp.nii.gz")

# -----------------------------
# 4) 误差分析
err = comp
err_mag = np.sqrt(np.sum(err ** 2, axis=-1))
print("Simulated composition error stats:")
print(" min =", np.min(err_mag))
print(" median =", np.median(err_mag))
print(" max =", np.max(err_mag))

# -----------------------------
# 5) 可视化几个切片
plt.figure(figsize=(12, 4))
for i, z in enumerate([Z // 4, Z // 2, 3 * Z // 4]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(err_mag[z], cmap='hot')
    plt.colorbar()
    plt.title(f"Error slice z={z}")
    plt.axis('off')
plt.suptitle("Simulated composition error magnitude")
plt.show()
