import shutil

import SimpleITK as sitk

# ----------------------------
# 1. 读取固定图像和移动图像
# ----------------------------
fixed_path = r"D:\Data\mir\validation\mask_cube.nii.gz"
moving_path = r"D:\Data\mir\validation\mask_cylinder.nii.gz"
# fixed_path = r"D:\Data\mir\validation\fixed.nii.gz"
# moving_path = r"D:\Data\mir\validation\moving.nii.gz"

fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

shutil.copy(fixed_path, "fixed_image.nii.gz")
shutil.copy(moving_path, "moving_image.nii.gz")

# ----------------------------
# 2. 初始化 Diffeomorphic Demons 注册器
# ----------------------------
demons = sitk.DiffeomorphicDemonsRegistrationFilter()
demons.SetNumberOfIterations(200)  # 迭代次数
demons.SetStandardDeviations(1.0)  # 平滑参数

# ----------------------------
# 3. 执行注册，获得位移场
# ----------------------------
displacement_field = demons.Execute(fixed, moving)
print("displacement_field size:", displacement_field.GetSize())
print("displacement_field dimension:", displacement_field.GetDimension())
print("displacement_field pixel type:", displacement_field.GetPixelIDTypeAsString())
sitk.WriteImage(displacement_field, "displacement_field.nii.gz")

# 将位移场转换为 Transform 对象，便于后续应用
displacement_transform = sitk.DisplacementFieldTransform(displacement_field)

# ----------------------------
# 4. 使用位移场重采样移动图像
# ----------------------------
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(displacement_transform)
resampler.SetDefaultPixelValue(0.0)

moved_image = resampler.Execute(moving)

print("moved_image size:", moved_image.GetSize())

# ----------------------------
# 5. 保存结果
# ----------------------------
sitk.WriteImage(moved_image, "moving_registered.nii.gz")

print("配准完成，结果已保存。")
