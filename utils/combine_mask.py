import os
import SimpleITK as sitk
import numpy as np

# 输入目录和输出目录
src = r"D:\Data\seg\open_atlas\test_atlas\LCTSC-Test-S2-201\STRUCTURES"
dst = r"D:\Data\seg\open_atlas\test_atlas\LCTSC-Test-S2-201\IMAGES"

# 获取所有的nii.gz结构文件
segs = [f for f in os.listdir(src) if f.endswith(".nii.gz")]
print("找到的结构文件:", segs)

# class_dic 及其反转
class_dic = {
    "1": "Heart",
    "2": "Ventricle_L",
    "3": "Ventricle_R",
    "4": "Atrium_L",
    "5": "Atrium_R",
    "6": "A_Aorta",
    "7": "A_Pulmonary",
    "8": "A_LAD",
    "9": "A_Cflx",
    "10": "A_Coronary_L",
    "11": "A_Coronary_R",
    "12": "V_Venacava_S",
    "13": "Valve_Mitral",
    "14": "Valve_Tricuspid",
    "15": "Valve_Aortic",
    "16": "Valve_Pulmonic",
    "17": "CN_Sinoatrial",
    "18": "CN_Atrioventricular"
}

# 反转字典，用于从结构名称查找标签值
reversed_class_dic = {v: int(k) for k, v in class_dic.items()}
print("反转字典:", reversed_class_dic)

# 读取第一个结构文件来获取图像尺寸和空间信息
first_seg_path = os.path.join(src, segs[0])
first_image = sitk.ReadImage(first_seg_path)

# 将 combined_array 数据类型设置为 float32，避免溢出
combined_array = np.zeros(sitk.GetArrayFromImage(first_image).shape, dtype=np.float32)

# 遍历每个结构文件，将它们合成到一个多标签数组中（累加处理）
for seg in segs:
    seg_name = seg.split(".")[0]  # 获取文件名，去掉后缀
    if seg_name=="Heart":
        continue
    label = reversed_class_dic.get(seg_name, None)  # 获取标签值
    if label is None:
        print(f"跳过未知结构: {seg_name}")
        continue

    print(f"处理 {seg_name}, 标签值: {label}")
    seg_path = os.path.join(src, seg)
    image = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(image)  # 将图像转换为 numpy 数组

    # 假设每个 nii.gz 是二值图，合成时用标签值累加
    combined_array[arr > 0] += label

# 转换回 SimpleITK 图像
combined_image = sitk.GetImageFromArray(combined_array)
combined_image.CopyInformation(first_image)  # 保留原始图像的空间信息

# 保存最终合成的多标签图像
output_path = os.path.join(dst, "mask.nii.gz")
sitk.WriteImage(combined_image, output_path)

print(f"合并后的多标签图像已保存: {output_path}")
