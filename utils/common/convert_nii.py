import SimpleITK as sitk

src_path = r"D:\code\Medical-Image-Registration\utils\data\atlas\LCTSC-Test-S2-201\IMAGES\CT.nii.gz"
dst_path = r"D:\code\Medical-Image-Registration\utils\data\atlas\LCTSC-Test-S2-201\IMAGES\CT2.nii.gz"

src = sitk.ReadImage(src_path)

dst = sitk.Cast(src, sitk.sitkInt16)

sitk.WriteImage(dst, dst_path)
