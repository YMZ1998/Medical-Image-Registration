import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# 读入位移场
disp = sitk.ReadImage(r"D:\debug\deformation_field.nii.gz", sitk.sitkVectorFloat64)

# 计算雅可比行列式
jacobian_det = sitk.DisplacementFieldJacobianDeterminant(disp)

# 转 numpy 可视化（二维示例）
jacobian_np = sitk.GetArrayFromImage(jacobian_det)
# jacobian_np = np.where(jacobian_np < 1, jacobian_np, 1)
# jacobian_np = np.where(jacobian_np > 0, jacobian_np, 0)
# 假设三维数组 jacobian_np.shape = (D,H,W)
D, H, W = jacobian_np.shape

# 初始切片索引
init_slice = D // 2

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # 给滑块留空间

img = ax.imshow(jacobian_np[init_slice, :, :], cmap='jet')
ax.set_title(f"Slice {init_slice}")
plt.colorbar(img, ax=ax)

# 添加滑块
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, 'Slice', 0, D-1, valinit=init_slice, valstep=1)

# 滑块事件
def update(val):
    slice_idx = int(slider.val)
    img.set_data(jacobian_np[slice_idx, :, :])
    ax.set_title(f"Slice {slice_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()


