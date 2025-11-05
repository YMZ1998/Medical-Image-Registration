import numpy as np
from PIL import Image

# 图像尺寸
width, height = 128, 128

# 生成一个三通道图像，三通道的像素值是坐标的累加值
image_array = np.zeros((height, width, 3), dtype=int)

# 对于每个通道，像素值是 (x + y)，可以加入一些通道偏移
for c in range(3):
    image_array[..., c] = np.fromfunction(lambda x, y: x + y + c * 50, (height, width), dtype=int)

# 将像素值限制在 0-255 范围内
image_array = np.clip(image_array, 0, 255)

# 将数组转换为图像
image = Image.fromarray(image_array.astype(np.uint8))

# 保存为 PNG 文件
image.save("./test_data/test.png")

# 显示图像（可选）
image.show()
