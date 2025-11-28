from totalsegmentator.python_api import totalsegmentator

from utils.dir_process import remove_and_create_dir

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

"TotalSegmentator -i D:\\Data\\seg\\open_atlas\\test_atlas\\LCTSC-Test-S2-201\\IMAGES\\CT.nii.gz -o D:\\Data\\seg\\open_atlas\\test_atlas\\LCTSC-Test-S2-201\\IMAGES\\segmentations  --device gpu"
"TotalSegmentator -i D:\\debug\\fixed_vol.nii.gz -o D:\\debug\\segmentations  --device gpu"
"TotalSegmentator -i C:\\Users\\Admin\\Desktop\\51144787_fb0fe5fa^db9de051 -o D:\\debug\\segmentations2  --device gpu"

if __name__ == "__main__":
    # input_path = r"D:\\debug\\test.nii.gz"
    # input_path = r"C:\Users\Admin\Desktop\17-2320_JIANGTAO"
    input_path = r"D:\debug\LUNG1-226\IMAGES\CT"
    output_path = r"D:\\debug\\segmentations"
    remove_and_create_dir(output_path)

    # option 1: provide input and output as file paths
    totalsegmentator(input_path, input_path, output_type="dicom", task="total",
                     device="gpu")
    # totalsegmentator(input_path, output_path, output_type="nifti",
    #                  device="gpu")
    # option 2: provide input and output as nifti image objects
    # input_img = nib.load(input_path)
    # nib.save(input_img, output_path)
    # output_img = totalsegmentator(input_img)
    # nib.save(output_img, output_path)
