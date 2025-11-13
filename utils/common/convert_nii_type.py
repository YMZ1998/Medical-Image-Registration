import SimpleITK as sitk

src_path = r"D:\Data\cbct\CT2\head1_seg.mhd"

src = sitk.ReadImage(src_path)

dst = sitk.Cast(src, sitk.sitkUInt8)

sitk.WriteImage(dst, src_path.replace(".mhd", ".nii.gz"))
sitk.WriteImage(dst, src_path)
