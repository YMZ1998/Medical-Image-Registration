import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


"TotalSegmentator -i D:\\Data\\seg\\open_atlas\\test_atlas\\LCTSC-Test-S2-201\\IMAGES\\CT.nii.gz -o D:\\Data\\seg\\open_atlas\\test_atlas\\LCTSC-Test-S2-201\\IMAGES\\segmentations  --device gpu"
