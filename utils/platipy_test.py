import platipy

from pathlib import Path
import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np
from matplotlib.pyplot import pause

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com
from platipy.imaging.utils.io import write_nrrd_structure_set
from platipy.imaging.utils.ventricle import generate_left_ventricle_segments

from platipy.imaging.projects.cardiac.run import install_open_atlas


atlas_path = Path("data/atlas")

if not atlas_path.exists():
    install_open_atlas(atlas_path)

patid = "LCTSC-Test-S2-201"

image_path = atlas_path.joinpath(patid, "IMAGES", "CT.nii.gz")
image = sitk.ReadImage(str(image_path)) # onyl used for visualisation

contours = {}

lv_path = atlas_path.joinpath(patid, "STRUCTURES", "Ventricle_L.nii.gz")
contours["Ventricle_L"] = sitk.ReadImage(str(lv_path))

la_path = atlas_path.joinpath(patid, "STRUCTURES", "Atrium_L.nii.gz")
contours["Atrium_L"] = sitk.ReadImage(str(la_path))

rv_path = atlas_path.joinpath(patid, "STRUCTURES", "Ventricle_R.nii.gz")
contours["Ventricle_R"] = sitk.ReadImage(str(rv_path))

heart_path = atlas_path.joinpath(patid, "STRUCTURES", "Heart.nii.gz")
contours["Heart"] = sitk.ReadImage(str(heart_path))

lv_segments = generate_left_ventricle_segments(contours, verbose=True)

vis = ImageVisualiser(image, cut=get_com(contours["Ventricle_L"]), figure_size_in=6)
vis.add_contour(contours)
vis.add_contour(lv_segments)
vis.set_limits_from_label(contours["Heart"], expansion=20)
fig = vis.show()

plt.show(block=True)

# write_nrrd_structure_set(lv_segments, atlas_path.joinpath(patid, "STRUCTURES", "LV_Segments.nrrd"), colormap=plt.cm.rainbow)