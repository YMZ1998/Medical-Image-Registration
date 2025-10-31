import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


disp1 = sitk.ReadImage("cum_AB_disp.nii.gz", sitk.sitkVectorFloat64)
disp2 = sitk.ReadImage("cum_BA_disp.nii.gz", sitk.sitkVectorFloat64)

# 计算雅可比行列式
jacobian1 = sitk.DisplacementFieldJacobianDeterminant(disp1)
jacobian2 = sitk.DisplacementFieldJacobianDeterminant(disp2)

# 转 numpy 数组 (D,H,W)
jacobian1_np = sitk.GetArrayFromImage(jacobian1)
jacobian2_np = sitk.GetArrayFromImage(jacobian2)
D, H, W = jacobian1_np.shape

# 初始切片
init_slice = D // 2

fig, axes = plt.subplots(1,2, figsize=(10,5))
plt.subplots_adjust(bottom=0.25)

# 初始显示
img1 = axes[0].imshow(jacobian1_np[init_slice,:,:], cmap='jet', vmin=jacobian1_np.min(), vmax=jacobian1_np.max())
axes[0].set_title(f"Disp1 Slice {init_slice}")
axes[0].axis('off')
img2 = axes[1].imshow(jacobian2_np[init_slice,:,:], cmap='jet', vmin=jacobian2_np.min(), vmax=jacobian2_np.max())
axes[1].set_title(f"Disp2 Slice {init_slice}")
axes[1].axis('off')
fig.colorbar(img1, ax=axes[0])
fig.colorbar(img2, ax=axes[1])

# 添加滑块
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Slice', 0, D-1, valinit=init_slice, valstep=1)

def update(val):
    slice_idx = int(slider.val)
    img1.set_data(jacobian1_np[slice_idx,:,:])
    axes[0].set_title(f"Disp1 Slice {slice_idx}")
    img2.set_data(jacobian2_np[slice_idx,:,:])
    axes[1].set_title(f"Disp2 Slice {slice_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
