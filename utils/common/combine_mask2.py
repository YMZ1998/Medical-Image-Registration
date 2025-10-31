import os
import SimpleITK as sitk
import numpy as np

from convert_to_hfs import convert_to_hfs

# 输入目录和输出目录
src = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030\segmentations"
dst = r"D:\Data\seg\Totalsegmentator_dataset_v201\s0030"

# 获取所有的nii.gz结构文件
segs = [f for f in os.listdir(src) if f.endswith(".nii.gz")]
print("找到的结构文件:", segs)

# 读取第一个结构文件来获取图像尺寸和空间信息
first_seg_path = os.path.join(src, segs[0])
first_image = sitk.ReadImage(first_seg_path)

# 将 combined_array 数据类型设置为 float32，避免溢出
combined_array = np.zeros(sitk.GetArrayFromImage(first_image).shape, dtype=np.float32)
dict = {}

bone_keywords = [
    "rib", "vertebrae", "skull", "clavicula", "femur", "hip",
    "humerus", "sacrum", "scapula", "sternum", "costal_cartilages"
]
# 遍历每个结构文件，将它们合成到一个多标签数组中（累加处理）
for label, seg in enumerate(segs):
    seg_name = seg.split(".")[0]  # 获取文件名，去掉后缀

    if not any(part in seg_name.lower() for part in bone_keywords):
        continue

    print(f"处理 {seg_name}, 标签值: {label}")
    dict[seg_name] = label
    seg_path = os.path.join(src, seg)
    try:
        image = sitk.ReadImage(seg_path)
    except:
        print(f"无法读取结构文件: {seg_path}")
        continue
    arr = sitk.GetArrayFromImage(image)

    combined_array[arr > 0] += label
print(dict)
combined_image = sitk.GetImageFromArray(combined_array)
combined_image.CopyInformation(first_image)  # 保留原始图像的空间信息

# 保存最终合成的多标签图像
output_path = os.path.join(dst, "mask.nii.gz")
sitk.WriteImage(combined_image, output_path)

print(f"合并后的多标签图像已保存: {output_path}")

convert_to_hfs(output_path, output_path)
