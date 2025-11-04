import os

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt


def resample_image(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    original_size = np.array(image.GetSize())
    target_size = np.array(target_size)
    original_spacing = np.array(image.GetSpacing())
    new_spacing = original_spacing * (original_size / target_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputSpacing([float(sp) for sp in new_spacing])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    return resampler.Execute(image)


def warp_image_with_ddf(fixed, moving, ddf_field):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    transform = sitk.DisplacementFieldTransform(ddf_field)
    resampler.SetTransform(transform)
    # print(transform)
    ddf = sitk.TransformToDisplacementField(
        transform,
        size=fixed.GetSize(),
        outputSpacing=fixed.GetSpacing(),
        outputOrigin=fixed.GetOrigin(),
        outputDirection=fixed.GetDirection()
    )
    sitk.WriteImage(ddf, r'D:\code\Medical-Image-Registration\installer\result\ddf.nii.gz')
    # print(type(transform))
    # sitk.WriteTransform(transform, r'D:\code\Medical-Image-Registration\installer\result\transform.txt')
    moved = resampler.Execute(moving)
    return moved


def vis_ddf(ddf_field):
    ddf = sitk.GetArrayFromImage(ddf_field)  # (Z, Y, X, 3)
    # 取中间层
    z = ddf.shape[0] // 2
    u = ddf[z, :, :, 0]
    v = ddf[z, :, :, 1]

    H, W = u.shape
    step = 5  # 网格间隔（像素）

    # 生成规则网格
    Y, X = np.mgrid[0:H:step, 0:W:step]

    # 形变后的网格点位置
    X_def = X + u[::step, ::step]
    Y_def = Y + v[::step, ::step]

    # 绘制网格
    plt.figure(figsize=(8, 8))
    for i in range(Y.shape[0]):
        plt.plot(X_def[i, :], Y_def[i, :], color='orange', lw=0.8)
    for j in range(X.shape[1]):
        plt.plot(X_def[:, j], Y_def[:, j], color='orange', lw=0.8)

    plt.gca().invert_yaxis()
    plt.title(f"Grid deformation at slice Z={z}")
    plt.axis("equal")
    plt.show()


def main():
    moving_image_path = "./data/moving.nii.gz"
    fixed_image_path = "./data/fixed.nii.gz"

    ddf_path = "./result/ddf_field.mhd"

    fixed_image = sitk.ReadImage(fixed_image_path)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    moving_image = sitk.ReadImage(moving_image_path)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # fixed_image = resample_image(fixed_image, (192, 192, 192))
    # moving_image = resample_image(moving_image, (192, 192, 192))

    ddf_field = sitk.ReadImage(ddf_path)
    ddf_field = sitk.Cast(ddf_field, sitk.sitkVectorFloat64)

    print("DDF spacing:", ddf_field.GetSpacing())
    print("Moving spacing:", moving_image.GetSpacing())
    print("DDF origin:", ddf_field.GetOrigin())
    print("Moving origin:", moving_image.GetOrigin())
    print("DDF direction:", ddf_field.GetDirection())
    print("Moving direction:", moving_image.GetDirection())
    print("DDF size:", ddf_field.GetSize())
    print("Moving image size:", moving_image.GetSize())

    # vis_ddf(ddf_field)

    arr = sitk.GetArrayFromImage(ddf_field)
    print("DDF min:", np.min(arr), "max:", np.max(arr), "mean:", np.mean(arr))

    print("DDF dimension:", ddf_field.GetDimension())
    print("DDF components per pixel:", ddf_field.GetNumberOfComponentsPerPixel())
    print("DDF pixel type:", ddf_field.GetPixelIDTypeAsString())

    # # voxel -> mm
    # spacing = np.array(moving_image.GetSpacing())  # (sx, sy, sz)
    # # 注意 numpy array 顺序是 (Z, Y, X), spacing 是 (X, Y, Z)
    # ddf_array_phys = np.zeros_like(arr)
    # ddf_array_phys[..., 0] = arr[..., 0] * spacing[0]  # X
    # ddf_array_phys[..., 1] = arr[..., 1] * spacing[1]  # Y
    # ddf_array_phys[..., 2] = arr[..., 2] * spacing[2]  # Z
    #
    # ddf_field_phys = sitk.GetImageFromArray(ddf_array_phys, isVector=True)
    # ddf_field_phys.CopyInformation(ddf_field)  # 保留 origin, direction, spacing

    moved_image = warp_image_with_ddf(fixed_image, moving_image, ddf_field)

    # moved_image = resample_image(moved_image, (224, 192, 224))

    output_path = os.path.join("./result", "moved_image.nii.gz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(moved_image, output_path)

    print(f"✅ Moved image saved to: {output_path}")


if __name__ == "__main__":
    main()
