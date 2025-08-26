import SimpleITK as sitk

image = sitk.ReadImage(r"D:\Data\Test\case2\Abdomen.nii.gz")
mask = sitk.ReadImage(r"D:\Data\Test\case2\Body.roi.nii.gz")

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(image)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetTransform(sitk.Transform())
resampler.SetDefaultPixelValue(0)

resampled_mask = resampler.Execute(mask)

sitk.WriteImage(resampled_mask, r"D:\Data\Test\case2\resampled_mask.nii.gz")
