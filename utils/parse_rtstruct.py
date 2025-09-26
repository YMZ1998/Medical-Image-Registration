import os
import json
import pydicom

from rtstruct_to_nii import find_rtstruct_dcm


def get_rois_and_colors(rtstruct_path):
    ds = pydicom.dcmread(rtstruct_path, force=True)
    # Build map ROINumber -> ROIName from StructureSetROISequence
    roi_name_map = {}
    ssrs = getattr(ds, "StructureSetROISequence", None)
    if ssrs is not None:
        for item in ssrs:
            rn = getattr(item, "ROINumber", None)
            name = getattr(item, "ROIName", None)
            # Some files may have ROIDisplayColor here as well
            color = getattr(item, "ROIDisplayColor", None)
            if rn is not None:
                roi_name_map[int(rn)] = {
                    "name": str(name) if name is not None else "",
                    "color_from_ssr": list(color) if color is not None else None
                }

    # Next, walk ROIContourSequence to get colors there (preferred if present)
    roi_contour_seq = getattr(ds, "ROIContourSequence", None)
    results = []
    if roi_contour_seq is not None:
        for item in roi_contour_seq:
            ref_num = getattr(item, "ReferencedROINumber", None)
            color = getattr(item, "ROIDisplayColor", None)
            # Some RTSTRUCTs store color as list of 3 ints, others may not include it
            color_list = list(color) if color is not None else None

            # Lookup name from map; fallback to empty string
            name = ""
            color_from_ssr = None
            if ref_num is not None and int(ref_num) in roi_name_map:
                name = roi_name_map[int(ref_num)]["name"]
                color_from_ssr = roi_name_map[int(ref_num)].get("color_from_ssr")

            # If no color in ROIContourSequence, fall back to StructureSetROISequence color
            final_color = color_list if color_list is not None else color_from_ssr

            results.append({
                "roi_number": int(ref_num) if ref_num is not None else None,
                "roi_name": name,
                "color": final_color
            })

    else:
        # Fallback: if there's no ROIContourSequence, use StructureSetROISequence info only
        for rn, info in roi_name_map.items():
            results.append({
                "roi_number": int(rn),
                "roi_name": info.get("name", ""),
                "color": info.get("color_from_ssr")
            })

    # Ensure we have a stable order (by roi_number when possible)
    results_sorted = sorted(results, key=lambda x: (x["roi_number"] is None, x["roi_number"]))
    return results_sorted


def main():
    ct_dir = r"C:\Users\Admin\Desktop\17-2320_JIANGTAO"
    RTSTRUCT_DCM = find_rtstruct_dcm(ct_dir)
    if not os.path.exists(RTSTRUCT_DCM):
        print(f"RTSTRUCT file not found: {RTSTRUCT_DCM}")
        return

    rois = get_rois_and_colors(RTSTRUCT_DCM)

    # Print pretty table
    print("\nExtracted ROIs and colors:")
    print("{:>6}  {:30}  {:15}".format("ROINr", "ROIName", "Color (R,G,B)"))
    print("-" * 60)
    for r in rois:
        rn = r["roi_number"] if r["roi_number"] is not None else "-"
        name = r["roi_name"] if r["roi_name"] else "-"
        color = r["color"] if r["color"] is not None else "-"
        print(f"{rn:>6}  {name:30}  {str(color):15}")

    # Save JSON next to RTSTRUCT
    out_dir = os.path.dirname(os.path.abspath(RTSTRUCT_DCM))
    out_json = os.path.join(out_dir, "rois_colors.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rois, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON -> {out_json}")


if __name__ == "__main__":
    main()
