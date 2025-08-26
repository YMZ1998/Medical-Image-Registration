import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from alternating_bspline_registration import sitk_to_numpy_disp


def compose_disp(u1, u2):
    # u1,u2: (Z,Y,X,3) index-space displacements (voxels)
    Z, Y, X, _ = u1.shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    phi_z = zz + u1[..., 0]
    phi_y = yy + u1[..., 1]
    phi_x = xx + u1[..., 2]
    coords = [phi_z, phi_y, phi_x]
    u2z = map_coordinates(u2[..., 0], coords, order=1, mode='nearest')
    u2y = map_coordinates(u2[..., 1], coords, order=1, mode='nearest')
    u2x = map_coordinates(u2[..., 2], coords, order=1, mode='nearest')
    comp = np.stack([u1[..., 0] + u2z,
                     u1[..., 1] + u2y,
                     u1[..., 2] + u2x], axis=-1)
    return comp


u_AB = sitk.ReadImage("cum_AB_disp.nii.gz", sitk.sitkVectorFloat64)
u_BA = sitk.ReadImage("cum_BA_disp.nii.gz", sitk.sitkVectorFloat64)
u_AB = sitk_to_numpy_disp(u_AB)
u_BA = sitk_to_numpy_disp(u_BA)
# Compose BA ∘ AB, then error to identity
comp = compose_disp(u_AB, u_BA)  # comp maps A -> A (should be identity)
# compute error displacement relative to identity (should be near zero)
err = comp  # because comp = displacement from x to comp(x) = x + err, so err ~ 0
max_err = np.max(np.abs(err))
mean_err = np.mean(np.sqrt(np.sum(err ** 2, axis=-1)))
print("composition max abs error (voxels):", max_err)
print("composition mean L2 error (voxels):", mean_err)

# 1) 统计误差分布
err_mag = np.sqrt(np.sum(err ** 2, axis=-1))
print("err magnitude stats (vox):")
print(" min =", np.min(err_mag))
print(" 5% =", np.percentile(err_mag, 5))
print(" median =", np.median(err_mag))
print(" 95% =", np.percentile(err_mag, 95))
print(" max =", np.max(err_mag))

# 2) 可视化几个切片
Z = err_mag.shape[0]
plt.figure(figsize=(12, 4))
for i, z in enumerate([Z // 4, Z // 2, 3 * Z // 4]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(err_mag[z], cmap="hot")
    plt.colorbar()
    plt.title(f"Error slice z={z}")
    plt.axis("off")
plt.suptitle("Composition error magnitude (voxels)")
plt.show()

# 3) 验证反向组合 AB∘BA
comp2 = compose_disp(u_BA, u_AB)  # comp2: B->B
err2 = comp2
err2_mag = np.sqrt(np.sum(err2 ** 2, axis=-1))
print("reverse composition stats:")
print(" median =", np.median(err2_mag), "max =", np.max(err2_mag))


# 4) Jacobian determinant check (局部可逆性)
# detJ ~ 1 if transform is diffeomorphic (no folding)
def jacobian_det(disp):
    # disp: (Z,Y,X,3) voxel-space displacement
    Z, Y, X, _ = disp.shape
    # 网格坐标
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    phi = np.stack([zz + disp[..., 0],
                    yy + disp[..., 1],
                    xx + disp[..., 2]], axis=-1)
    # 中心差分计算梯度
    grad = np.gradient(phi[..., 0]), np.gradient(phi[..., 1]), np.gradient(phi[..., 2])
    detJ = (grad[0][0] * (grad[1][1] * grad[2][2] - grad[1][2] * grad[2][1])
            - grad[0][1] * (grad[1][0] * grad[2][2] - grad[1][2] * grad[2][0])
            + grad[0][2] * (grad[1][0] * grad[2][1] - grad[1][1] * grad[2][0]))
    return detJ


detJ_AB = jacobian_det(u_AB)
print("Jacobian determinant stats AB: min =", np.min(detJ_AB), "median =", np.median(detJ_AB))
