import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# 读取位移场
disp1 = sitk.ReadImage("sim_u_AB.nii.gz", sitk.sitkVectorFloat64)
disp2 = sitk.ReadImage("sim_u_BA.nii.gz", sitk.sitkVectorFloat64)
comp = sitk.ReadImage("sim_comp.nii.gz", sitk.sitkVectorFloat64)

# 计算雅可比行列式
jacobian1 = sitk.DisplacementFieldJacobianDeterminant(disp1)
jacobian2 = sitk.DisplacementFieldJacobianDeterminant(disp2)

# 转 numpy 数组 (D,H,W)
jacobian1_np = sitk.GetArrayFromImage(jacobian1)
jacobian2_np = sitk.GetArrayFromImage(jacobian2)
comp_np = sitk.GetArrayFromImage(sitk.VectorMagnitude(comp))  # 误差幅值
D, H, W = jacobian1_np.shape

# 初始切片
init_slice = D // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25)

# 初始显示
img1 = axes[0].imshow(jacobian1_np[init_slice, :, :], cmap='jet', vmin=jacobian1_np.min(), vmax=jacobian1_np.max())
axes[0].set_title(f"Disp1 Slice {init_slice}")
axes[0].axis('off')

img2 = axes[1].imshow(jacobian2_np[init_slice, :, :], cmap='jet', vmin=jacobian2_np.min(), vmax=jacobian2_np.max())
axes[1].set_title(f"Disp2 Slice {init_slice}")
axes[1].axis('off')

img3 = axes[2].imshow(comp_np[init_slice, :, :], cmap='jet', vmin=min(comp_np.min(), 0), vmax=max(comp_np.max(), 1.0))
axes[2].set_title(f"Comp Error Slice {init_slice}")
axes[2].axis('off')

fig.colorbar(img1, ax=axes[0])
fig.colorbar(img2, ax=axes[1])
fig.colorbar(img3, ax=axes[2])

# 添加滑块
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Slice', 0, D - 1, valinit=init_slice, valstep=1)


def update(val):
    slice_idx = int(slider.val)
    img1.set_data(jacobian1_np[slice_idx, :, :])
    axes[0].set_title(f"Disp1 Slice {slice_idx}")

    img2.set_data(jacobian2_np[slice_idx, :, :])
    axes[1].set_title(f"Disp2 Slice {slice_idx}")

    img3.set_data(comp_np[slice_idx, :, :])
    axes[2].set_title(f"Comp Error Slice {slice_idx}")

    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
