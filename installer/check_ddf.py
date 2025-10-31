import SimpleITK as sitk
import os
import numpy as np


def warp_image_with_ddf(original_image, ddf_field):
    inv_ddf = sitk.InvertDisplacementField(
        ddf_field,
        maximumNumberOfIterations=50,
        maxErrorToleranceThreshold=0.1,
        meanErrorToleranceThreshold=0.001,
        enforceBoundaryCondition=True
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetTransform(sitk.DisplacementFieldTransform(inv_ddf))
    moved = resampler.Execute(original_image)
    return moved


def main():
    moving_image_path = "./data/moving.nii.gz"
    ddf_path = "./result/ddf_field.mhd"

    moving_image = sitk.ReadImage(moving_image_path)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    ddf_field = sitk.ReadImage(ddf_path)
    ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    print("DDF spacing:", ddf_field.GetSpacing())
    print("Moving spacing:", moving_image.GetSpacing())
    print("DDF origin:", ddf_field.GetOrigin())
    print("Moving origin:", moving_image.GetOrigin())
    print("DDF direction:", ddf_field.GetDirection())
    print("Moving direction:", moving_image.GetDirection())

    arr = sitk.GetArrayFromImage(ddf_field)
    print("DDF min:", np.min(arr), "max:", np.max(arr), "mean:", np.mean(arr))
    print("Approx displacement (mm):", np.linalg.norm([np.max(arr[..., 0]), np.max(arr[..., 1]), np.max(arr[..., 2])]))

    print("DDF size:", ddf_field.GetSize())
    print("Moving image size:", moving_image.GetSize())

    print("DDF dimension:", ddf_field.GetDimension())
    print("DDF components per pixel:", ddf_field.GetNumberOfComponentsPerPixel())
    print("DDF pixel type:", ddf_field.GetPixelIDTypeAsString())

    if ddf_field.GetSize() != moving_image.GetSize():
        print("⚠️ Warning: DDF size does not match moving image size! Resampling DDF...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_image)
        resampler.SetOutputSpacing(moving_image.GetSpacing())
        resampler.SetOutputOrigin(moving_image.GetOrigin())
        resampler.SetOutputDirection(moving_image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        ddf_field = resampler.Execute(ddf_field)
        print("DDF resampled size:", ddf_field.GetSize())

    moved_image = warp_image_with_ddf(moving_image, ddf_field)

    output_path = os.path.join("./result", "moved_image.nii.gz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(moved_image, output_path)

    print(f"✅ Moved image saved to: {output_path}")


if __name__ == "__main__":
    main()
