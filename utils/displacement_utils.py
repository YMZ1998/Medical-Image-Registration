import SimpleITK as sitk


def compose_displacement_fields(disp1: sitk.Image,
                                disp2: sitk.Image,
                                reference: sitk.Image = None) -> sitk.Image:
    """
    Compose two displacement fields: result = disp2 ∘ disp1
    (apply disp1 first, then disp2).

    Parameters
    ----------
    disp1 : sitk.Image
        First displacement field (Vector image).
    disp2 : sitk.Image
        Second displacement field (Vector image).
    reference : sitk.Image, optional
        Reference image defining size/spacing/origin/direction of output.
        If None or dimension mismatch, use disp1 as reference.

    Returns
    -------
    sitk.Image
        Composed displacement field.
    """

    # 确保类型为 VectorFloat64
    if disp1.GetPixelID() != sitk.sitkVectorFloat64:
        disp1 = sitk.Cast(disp1, sitk.sitkVectorFloat64)
    if disp2.GetPixelID() != sitk.sitkVectorFloat64:
        disp2 = sitk.Cast(disp2, sitk.sitkVectorFloat64)

    # fallback reference
    ref_image = reference
    if ref_image is None:
        ref_image = sitk.Image(disp1)

    # 维度检查
    dim1, dim2 = disp1.GetDimension(), disp2.GetDimension()
    print(f"Dimension : disp1={dim1}D, disp2={dim2}D")
    print("disp1:", disp1.GetDimension(), disp1.GetNumberOfComponentsPerPixel(), disp1.GetPixelIDTypeAsString())
    print("disp2:", disp2.GetDimension(), disp2.GetNumberOfComponentsPerPixel(), disp2.GetPixelIDTypeAsString())
    # size=disp1.GetSize()
    # origin=disp1.GetOrigin()
    # spacing=disp1.GetSpacing()
    # direction=disp1.GetDirection()
    dim = dim1
    tx1 = sitk.DisplacementFieldTransform(disp1)
    tx2 = sitk.DisplacementFieldTransform(disp2)

    composite = sitk.CompositeTransform(dim)
    composite.AddTransform(tx1)
    composite.AddTransform(tx2)

    # disp_composed = sitk.TransformToDisplacementField(
    #     composite,
    #     sitk.sitkVectorFloat64,
    #     size,
    #     origin,
    #     spacing,
    #     direction
    # )
    disp_composed = sitk.TransformToDisplacementField(
        composite,
        sitk.sitkVectorFloat64,
        ref_image.GetSize(),
        ref_image.GetOrigin(),
        ref_image.GetSpacing(),
        ref_image.GetDirection()
    )

    return disp_composed


def invert_displacement_field(disp: sitk.Image,
                              reference: sitk.Image = None,
                              max_iter: int = 50,
                              tol: float = 1e-6) -> sitk.Image:
    """
    Compute the inverse of a displacement field using SimpleITK.

    Parameters
    ----------
    disp : sitk.Image
        Input displacement field (vector image).
    reference : sitk.Image, optional
        Reference image defining size/spacing/origin/direction of output.
        If None, use disp itself as reference.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    sitk.Image
        Approximate inverse displacement field.
    """
    if reference is None:
        reference = sitk.Image(disp)

    inv_disp = sitk.InverseDisplacementField(
        disp,
        outputSpacing=reference.GetSpacing(),
        outputOrigin=reference.GetOrigin(),
        size=reference.GetSize(),
        subsamplingFactor=8  # 可调

    )

    return inv_disp


if __name__ == '__main__':
    # 读入两个位移场
    # disp1 = sitk.ReadImage("cum_AB_disp.nii.gz", sitk.sitkVectorFloat64)
    # disp2 = sitk.ReadImage("cum_BA_disp.nii.gz", sitk.sitkVectorFloat64)
    disp1 = sitk.ReadImage("sim_u_AB.nii.gz", sitk.sitkVectorFloat64)
    disp2 = sitk.ReadImage("sim_u_BA.nii.gz", sitk.sitkVectorFloat64)
    # 求逆
    inv_disp1 = invert_displacement_field(disp1)
    sitk.WriteImage(inv_disp1, "disp1_inverse.nii.gz")
    # 合成
    disp_composed = compose_displacement_fields(disp1, disp2)
    sitk.WriteImage(disp_composed, "disp_composed.nii.gz")


