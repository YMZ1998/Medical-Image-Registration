import SimpleITK as sitk
import os

def warp_image_with_ddf(original_image, ddf_field):
    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(original_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetTransform(sitk.DisplacementFieldTransform(ddf_field))  # 使用 DDF 变换

    moved = resampler.Execute(original_image)
    return moved

def main():
    moving_image_path = "../results/seg_resnet/moving_image.nii.gz"
    ddf_path = "../results/ddf_field.nii.gz"

    moving_image = sitk.ReadImage(moving_image_path)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    ddf_field = sitk.ReadImage(ddf_path, sitk.sitkVectorFloat64)

    if ddf_field.GetSize() != moving_image.GetSize():
        print("Warning: DDF size does not match moving image size!")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_image)
        resampler.SetSize(moving_image.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        ddf_field_resampled = resampler.Execute(ddf_field)
        ddf_field = ddf_field_resampled
        print("DDF Resampled size:", ddf_field.GetSize())

    moved_image = warp_image_with_ddf(moving_image, ddf_field)

    output_path = os.path.join("../results", "moved_image.nii.gz")
    sitk.WriteImage(moved_image, output_path)

    print(f"Moved image saved to: {output_path}")

if __name__ == "__main__":
    main()
