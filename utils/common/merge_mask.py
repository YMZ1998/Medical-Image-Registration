import os
import SimpleITK as sitk
import numpy as np

from convert_to_hfs import convert_to_hfs

src = r"D:\\debug\\segmentations"
dst = r"D:\\debug"

segs = [f for f in os.listdir(src) if f.endswith(".nii.gz")][:10]
segs.sort()  # 一定排序，确保标签稳定

print("找到的结构文件:", segs)

first_seg_path = os.path.join(src, segs[0])
first_image = sitk.ReadImage(first_seg_path)
combined_array = np.zeros(sitk.GetArrayFromImage(first_image).shape, dtype=np.uint16)

label_map = {}  # 原 dict 改名

for label, seg in enumerate(segs, start=1):
    seg_name = seg.replace(".nii.gz", "")
    seg_path = os.path.join(src, seg)

    print(f"处理: {seg_name}, 标签: {label}")
    label_map[seg_name] = label

    image = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(image)

    combined_array[arr > 0] = label  # 覆盖方式替代累加

print("标签映射:", label_map)

combined_image = sitk.GetImageFromArray(combined_array)
combined_image.CopyInformation(first_image)

output_path = os.path.join(dst, "test_mask.nii.gz")
sitk.WriteImage(combined_image, output_path)

print(f"合并的多标签图像输出: {output_path}")

# convert_to_hfs(output_path, output_path)
