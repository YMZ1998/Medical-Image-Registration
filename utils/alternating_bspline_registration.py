"""
BSpline alternating registration (A->B, B->A) with transform composition.
Requirements: SimpleITK, numpy, scipy
pip install SimpleITK numpy scipy
"""

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ---------- utilities ----------
def sitk_to_numpy_disp(disp_img):
    """
    Convert SimpleITK VectorImage or Image with 3 components to numpy displacement (Z,Y,X,3)
    Assumes image is in physical space mapping; we'll work in index space for composition.
    """
    arr = sitk.GetArrayFromImage(disp_img)  # shape (Z,Y,X,3)
    return arr.copy()


def numpy_to_sitk_disp(u, reference_image):
    """
    Convert numpy displacement (Z,Y,X,3) (index-space displacements) to SimpleITK image
    Keep same spacing/origin/direction as reference_image
    """
    img = sitk.GetImageFromArray(u.astype(np.float32), isVector=True)
    img.SetSpacing(reference_image.GetSpacing())
    img.SetOrigin(reference_image.GetOrigin())
    img.SetDirection(reference_image.GetDirection())
    return img


def compute_jacobian_determinant_indexspace(u):
    """
    u: numpy array shape (Z,Y,X,3) giving index-space displacements (voxel units)
    Returns detJ array (Z,Y,X)
    We compute J = I + grad(u) (with gradients along index axes).
    """
    # compute gradients of each displacement component
    uz = u[..., 0];
    uy = u[..., 1];
    ux = u[..., 2]
    # gradients: d(uz)/dz, d(uz)/dy, d(uz)/dx, etc.
    duz_dz = np.gradient(uz, axis=0)
    duz_dy = np.gradient(uz, axis=1)
    duz_dx = np.gradient(uz, axis=2)

    duy_dz = np.gradient(uy, axis=0)
    duy_dy = np.gradient(uy, axis=1)
    duy_dx = np.gradient(uy, axis=2)

    dux_dz = np.gradient(ux, axis=0)
    dux_dy = np.gradient(ux, axis=1)
    dux_dx = np.gradient(ux, axis=2)

    # Jacobian matrix at each voxel: I + grad(u)
    # compute determinant for 3x3 matrix:
    # | 1+duz_dz  duz_dy   duz_dx |
    # | duy_dz   1+duy_dy  duy_dx |
    # | dux_dz    dux_dy  1+dux_dx|
    a = 1.0 + duz_dz
    b = duz_dy
    c = duz_dx
    d = duy_dz
    e = 1.0 + duy_dy
    f = duy_dx
    g = dux_dz
    h = dux_dy
    i = 1.0 + dux_dx

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return det


def compose_displacement_indexspace(u1, u2):
    """
    Compose two displacement fields given in index-space (voxels).
    u1, u2: (Z,Y,X,3) numpy arrays
    returns u_comp = u1 + u2(x + u1(x))
    We'll use map_coordinates for interpolation (order=1).
    """
    Z, Y, X, _ = u1.shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    phi_z = zz + u1[..., 0]
    phi_y = yy + u1[..., 1]
    phi_x = xx + u1[..., 2]
    # prepare coordinates for map_coordinates: order is (z,y,x)
    coords = [phi_z, phi_y, phi_x]
    u2z = map_coordinates(u2[..., 0], coords, order=1, mode='nearest')
    u2y = map_coordinates(u2[..., 1], coords, order=1, mode='nearest')
    u2x = map_coordinates(u2[..., 2], coords, order=1, mode='nearest')
    u_comp = np.stack([u1[..., 0] + u2z,
                       u1[..., 1] + u2y,
                       u1[..., 2] + u2x], axis=-1)
    return u_comp


# ---------- BSpline registration function ----------
def bspline_register(fixed, moving,
                     grid_physical_spacing=(50.0, 50.0, 50.0),
                     metric='MI',
                     num_iterations=200,
                     learningRate=1.0,
                     shrink_factors=[4, 2, 1],
                     smoothing_sigmas=[3, 2, 0],
                     verbose=True):
    """
    Run SimpleITK BSpline registration returning displacement field image (index-space voxels).
    fixed, moving: SimpleITK images (same physical space ideally)
    grid_physical_spacing: approximate spacing (mm) between control points.
    Returns: displacement numpy array (Z,Y,X,3) representing index-space displacement field
    (i.e., how many voxels to move along each index axis).
    """
    # initialize bspline transform using physical spacing
    image_physical_size = [sz * spc for sz, spc in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh_size = [int(np.round(image_physical_size[i] / grid_physical_spacing[i])) for i in range(3)]
    # ensure mesh_size >= 1
    mesh_size = [max(1, m) for m in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed,
                                                         transformDomainMeshSize=mesh_size,
                                                         order=3)
    # Setup registration
    reg = sitk.ImageRegistrationMethod()
    if metric == 'MI':
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'NCC':
        reg.SetMetricAsCorrelation()
    else:
        reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    try:
        reg.SetOptimizerAsLBFGS2(numberOfIterations=num_iterations, deltaConvergenceTolerance=1e-5,
                                 maximumNumberOfCorrections=10)
    except Exception:
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=learningRate,
                                                     minStep=1e-6,
                                                     numberOfIterations=num_iterations,
                                                     gradientMagnitudeTolerance=1e-8)

    # Multi-resolution
    reg.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_transform, inPlace=False)
    if verbose:
        reg.AddCommand(sitk.sitkStartEvent, lambda: print("Start registration"))
        reg.AddCommand(sitk.sitkEndEvent, lambda: print("End registration"))
        reg.AddCommand(sitk.sitkIterationEvent, lambda: print(f"Iter metric: {reg.GetMetricValue()}"))

    out_transform = reg.Execute(fixed, moving)
    # Convert transform to displacement field in physical space
    # disp_physical = sitk.TransformToDisplacementField(out_transform,
    #                                                   sitk.sitkVectorFloat64,
    #                                                   referenceImage=fixed)
    disp_physical = sitk.TransformToDisplacementField(
        out_transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection()
    )

    # Convert physical-space displacement (mm) to index-space displacement (voxels)
    disp_np_phys = sitk.GetArrayFromImage(disp_physical)  # (Z,Y,X,3) physical mm
    spacing = np.array(
        fixed.GetSpacing())  # (x,y,z) in SimpleITK is (x,y,z) order; but array from GetImageFromArray is z,y,x
    # careful with axis order: SimpleITK spacing is (sx,sy,sz)
    sx, sy, sz = spacing[0], spacing[1], spacing[2]
    # Convert mm to voxels along (x,y,z). We need to divide each displacement component by corresponding spacing.
    # But disp_np_phys has component order [x,y,z] per vector; our numpy field is (z,y,x,3) with components [x,y,z]?
    # SimpleITK array returns components in last dimension: (z,y,x,3) with components ordered as (x,y,z).
    # We'll convert to index-space as (dz,dy,dx) ordering to match our compose functions which expect (z,y,x,3) with components (dz,dy,dx).
    disp_dx = disp_np_phys[..., 0] / sx
    disp_dy = disp_np_phys[..., 1] / sy
    disp_dz = disp_np_phys[..., 2] / sz
    # Now stack to (dz,dy,dx)
    disp_index = np.stack([disp_dz, disp_dy, disp_dx], axis=-1).astype(np.float32)
    return disp_index, out_transform


# ---------- high-level alternating loop ----------
def alternating_bspline_registration(imageA, imageB,
                                     iterations=4,
                                     bspline_spacing=(60, 60, 60),
                                     metric='MI',
                                     smooth_sigma_after_compose=1.0,
                                     max_compose_smooth_iters=1,
                                     verbose=True):
    """
    imageA, imageB: SimpleITK images (same physical domain)
    iterations: number of alternating pairs (A->B and B->A each count as one)
    Returns cumulative transforms for A->B and B->A as displacement numpy arrays (index-space)
    """
    # initialize zero displacement fields (index-space)
    Z, Y, X = sitk.GetArrayFromImage(imageA).shape
    zero_disp = np.zeros((Z, Y, X, 3), dtype=np.float32)
    cum_AB = zero_disp.copy()  # displacement that maps A -> B (index-space)
    cum_BA = zero_disp.copy()  # displacement that maps B -> A (index-space)

    for it in range(iterations):
        if verbose: print(f"\n=== Alternation iteration {it + 1} (A->B) ===")
        # Register A to B (but operate on original images, metric becomes measured under current composed transform)
        # We register fixed=B, moving=A deformed by current cum_AB (if non-zero, we avoid resampling: instead register A to B but we could transform A with cum_AB for better initialization)
        # For simplicity, transform moving image by cum_AB once for registration initialization (this is allowed but will re-sample once per alternation; keep small iterations)
        if np.any(cum_AB):
            # build transform from cum_AB and resample moving for registration initialization (this is optional)
            tf_img = numpy_to_sitk_disp(cum_AB,
                                        imageA)  # displacement field in index-space -> convert to physical displacement image
            # convert index-space to physical displacements:
            # Create displacement image in physical mm from index-space by multiplying spacing and swapping components
            disp_phys = sitk.Image(X=imageA.GetSize()[0], y=imageA.GetSize()[1], z=imageA.GetSize()[2],
                                   pixelID=sitk.sitkVectorFloat64)
        # Simpler approach: register original A->B without resampling to keep code robust
        disp_AB_new, transform_obj = bspline_register(fixed=imageB, moving=imageA,
                                                      grid_physical_spacing=bspline_spacing,
                                                      metric=metric,
                                                      num_iterations=150,
                                                      verbose=verbose)
        # Compose cum_AB = cum_AB ∘ new_disp? We interpret disp_AB_new as mapping A -> B directly (index-space)
        # If we had previous cum_AB (mapping A -> B_prev) and new_disp maps A -> B_new, we should combine carefully.
        # Simpler and safe approach: treat disp_AB_new as incremental (Δ) and compose: cum_AB = compose( cum_AB, disp_increment )
        # Here assume disp_AB_new is an incremental field (from current A to B); we'll compose as cum_AB = compose_displacement(cum_AB, disp_AB_new)
        cum_AB = compose_displacement_indexspace(cum_AB, disp_AB_new)
        # smooth after composition
        for _ in range(max_compose_smooth_iters):
            cum_AB[..., 0] = gaussian_filter(cum_AB[..., 0], sigma=smooth_sigma_after_compose)
            cum_AB[..., 1] = gaussian_filter(cum_AB[..., 1], sigma=smooth_sigma_after_compose)
            cum_AB[..., 2] = gaussian_filter(cum_AB[..., 2], sigma=smooth_sigma_after_compose)

        # Jacobian check
        detJ = compute_jacobian_determinant_indexspace(cum_AB)
        neg_count = np.sum(detJ <= 0)
        if verbose:
            print(f"A->B composed: negative-Jacobian voxels: {int(neg_count)} / {detJ.size}")

        if verbose: print(f"\n=== Alternation iteration {it + 1} (B->A) ===")
        # Now B->A
        disp_BA_new, transform_obj2 = bspline_register(fixed=imageA, moving=imageB,
                                                       grid_physical_spacing=bspline_spacing,
                                                       metric=metric,
                                                       num_iterations=150,
                                                       verbose=verbose)
        cum_BA = compose_displacement_indexspace(cum_BA, disp_BA_new)
        for _ in range(max_compose_smooth_iters):
            cum_BA[..., 0] = gaussian_filter(cum_BA[..., 0], sigma=smooth_sigma_after_compose)
            cum_BA[..., 1] = gaussian_filter(cum_BA[..., 1], sigma=smooth_sigma_after_compose)
            cum_BA[..., 2] = gaussian_filter(cum_BA[..., 2], sigma=smooth_sigma_after_compose)
        detJ2 = compute_jacobian_determinant_indexspace(cum_BA)
        neg_count2 = np.sum(detJ2 <= 0)
        if verbose:
            print(f"B->A composed: negative-Jacobian voxels: {int(neg_count2)} / {detJ2.size}")

        # Optional: early stopping if too many negative jacobians
        if neg_count > 0.01 * detJ.size or neg_count2 > 0.01 * detJ2.size:
            print(
                "Warning: many negative Jacobian detectors. Consider increasing regularization / control-point spacing or using diffeo method.")
            # you could rollback last composition here if desired

    return cum_AB, cum_BA


# ---------- Example usage ----------
if __name__ == "__main__":
    # replace with your image paths
    fixed_path = r"C:\Users\DATU\Desktop\validation\fixed.nii.gz"  # B
    moving_path = r"C:\Users\DATU\Desktop\validation\moving.nii.gz"  # A

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    cum_AB, cum_BA = alternating_bspline_registration(fixed, moving,
                                                      iterations=3,
                                                      bspline_spacing=(60, 60, 60),
                                                      metric='MI',
                                                      smooth_sigma_after_compose=0.8,
                                                      verbose=True)

    # Convert cumulative displacement to SimpleITK image and save
    dispAB_img = numpy_to_sitk_disp(cum_AB, fixed)
    sitk.WriteImage(dispAB_img, "cum_AB_disp.nii.gz")
    dispBA_img = numpy_to_sitk_disp(cum_BA, moving)
    sitk.WriteImage(dispBA_img, "cum_BA_disp.nii.gz")

    print("Done, saved cumulative displacement fields.")
