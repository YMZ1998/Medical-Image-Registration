import json
import os.path
from datetime import datetime

# 原始器官字典
class_dic2 = {'adrenal_gland_left': 0, 'adrenal_gland_right': 1, 'aorta': 2, 'atrial_appendage_left': 3,
             'autochthon_left': 4, 'autochthon_right': 5, 'brachiocephalic_trunk': 6, 'brachiocephalic_vein_left': 7,
             'brachiocephalic_vein_right': 8, 'brain': 9, 'clavicula_left': 10, 'clavicula_right': 11, 'colon': 12,
             'common_carotid_artery_left': 13, 'common_carotid_artery_right': 14, 'costal_cartilages': 15,
             'duodenum': 16, 'esophagus': 17, 'femur_left': 18, 'femur_right': 19, 'gallbladder': 20,
             'gluteus_maximus_left': 21, 'gluteus_maximus_right': 22, 'gluteus_medius_left': 23,
             'gluteus_medius_right': 24, 'gluteus_minimus_left': 25, 'gluteus_minimus_right': 26, 'heart': 27,
             'hip_left': 28, 'hip_right': 29, 'humerus_left': 30, 'humerus_right': 31, 'iliac_artery_left': 32,
             'iliac_artery_right': 33, 'iliac_vena_left': 34, 'iliac_vena_right': 35, 'iliopsoas_left': 36,
             'iliopsoas_right': 37, 'inferior_vena_cava': 38, 'kidney_cyst_left': 39, 'kidney_cyst_right': 40,
             'kidney_left': 41, 'kidney_right': 42, 'liver': 43, 'lung_lower_lobe_left': 44,
             'lung_lower_lobe_right': 45, 'lung_middle_lobe_right': 46, 'lung_upper_lobe_left': 47,
             'lung_upper_lobe_right': 48, 'pancreas': 49, 'portal_vein_and_splenic_vein': 50, 'prostate': 51,
             'pulmonary_vein': 52, 'rib_left_1': 53, 'rib_left_10': 54, 'rib_left_11': 55, 'rib_left_12': 56,
             'rib_left_2': 57, 'rib_left_3': 58, 'rib_left_4': 59, 'rib_left_5': 60, 'rib_left_6': 61, 'rib_left_7': 62,
             'rib_left_8': 63, 'rib_left_9': 64, 'rib_right_1': 65, 'rib_right_10': 66, 'rib_right_11': 67,
             'rib_right_12': 68, 'rib_right_2': 69, 'rib_right_3': 70, 'rib_right_4': 71, 'rib_right_5': 72,
             'rib_right_6': 73, 'rib_right_7': 74, 'rib_right_8': 75, 'rib_right_9': 76, 'sacrum': 77,
             'scapula_left': 78, 'scapula_right': 79, 'skull': 80, 'small_bowel': 81, 'spinal_cord': 82, 'spleen': 83,
             'sternum': 84, 'stomach': 85, 'subclavian_artery_left': 86, 'subclavian_artery_right': 87,
             'superior_vena_cava': 88, 'thyroid_gland': 89, 'trachea': 90, 'urinary_bladder': 91, 'vertebrae_C1': 92,
             'vertebrae_C2': 93, 'vertebrae_C3': 94, 'vertebrae_C4': 95, 'vertebrae_C5': 96, 'vertebrae_C6': 97,
             'vertebrae_C7': 98, 'vertebrae_L1': 99, 'vertebrae_L2': 100, 'vertebrae_L3': 101, 'vertebrae_L4': 102,
             'vertebrae_L5': 103, 'vertebrae_S1': 104, 'vertebrae_T1': 105, 'vertebrae_T10': 106, 'vertebrae_T11': 107,
             'vertebrae_T12': 108, 'vertebrae_T2': 109, 'vertebrae_T3': 110, 'vertebrae_T4': 111, 'vertebrae_T5': 112,
             'vertebrae_T6': 113, 'vertebrae_T7': 114, 'vertebrae_T8': 115, 'vertebrae_T9': 116
             }
class_dic = {'clavicula_left': 10, 'clavicula_right': 11, 'costal_cartilages': 15, 'femur_left': 18, 'femur_right': 19,
             'hip_left': 28, 'hip_right': 29, 'humerus_left': 30, 'humerus_right': 31, 'rib_left_1': 53,
             'rib_left_10': 54, 'rib_left_11': 55, 'rib_left_12': 56, 'rib_left_2': 57, 'rib_left_3': 58,
             'rib_left_4': 59, 'rib_left_5': 60, 'rib_left_6': 61, 'rib_left_7': 62, 'rib_left_8': 63, 'rib_left_9': 64,
             'rib_right_1': 65, 'rib_right_10': 66, 'rib_right_11': 67, 'rib_right_12': 68, 'rib_right_2': 69,
             'rib_right_3': 70, 'rib_right_4': 71, 'rib_right_5': 72, 'rib_right_6': 73, 'rib_right_7': 74,
             'rib_right_8': 75, 'rib_right_9': 76, 'sacrum': 77, 'scapula_left': 78, 'scapula_right': 79, 'skull': 80,
             'sternum': 84, 'vertebrae_C1': 92, 'vertebrae_C2': 93, 'vertebrae_C3': 94, 'vertebrae_C4': 95,
             'vertebrae_C5': 96, 'vertebrae_C6': 97, 'vertebrae_C7': 98, 'vertebrae_L1': 99, 'vertebrae_L2': 100,
             'vertebrae_L3': 101, 'vertebrae_L4': 102, 'vertebrae_L5': 103, 'vertebrae_S1': 104, 'vertebrae_T1': 105,
             'vertebrae_T10': 106, 'vertebrae_T11': 107, 'vertebrae_T12': 108, 'vertebrae_T2': 109, 'vertebrae_T3': 110,
             'vertebrae_T4': 111, 'vertebrae_T5': 112, 'vertebrae_T6': 113, 'vertebrae_T7': 114, 'vertebrae_T8': 115,
             'vertebrae_T9': 116}


# 生成 ROI 列表，每个器官随机分配颜色
def generate_color(i):
    # 简单生成不同颜色
    return [(i * 37) % 256, (i * 59) % 256, (i * 83) % 256]


roi_list = []
for name, idx in class_dic.items():
    roi_list.append({
        "roi_name": name,
        "color": generate_color(idx),
        "type": "ORGAN",
        "translation": name.replace("_", " ").title()
    })
template1 = {
    "task": "TotalMarrowIrradiation",
    "active": "True",
    "group": "OAR",
    "model_path": "./engine/dipper.ai.contour.oar.totalmarrow.unet3d.engine",
    "mean_intensity": 50.0,
    "std_intensity": 80.0,
    "lower_bound": -600.0,
    "upper_bound": 200.0,
    "current_spacing": [1.0, 1.0, 3.0],
    "patch_size": [224, 128, 64],
    "batch_size": 1,
    "step": 2,
    "translation": "全骨髓勾画",
    "creator": "Datu Medical AI Service",
    "date": datetime.now().strftime("%Y%m%d%H%M%S"),
    "description": "全骨髓放疗自动勾画",
    "class_dic": {str(v): k for k, v in class_dic.items()},
    "roi_list": roi_list
}

# 生成完整 JSON
template = {
    "task": "FullBodySegmentation",
    "active": "True",
    "group": "OAR",
    "model_path": "./engine/dipper.ai.contour.oar.fullbody.unet3d.engine",
    "mean_intensity": 50.0,
    "std_intensity": 80.0,
    "lower_bound": -1000.0,
    "upper_bound": 1000.0,
    "current_spacing": [1.0, 1.0, 3.0],
    "patch_size": [128, 128, 64],
    "batch_size": 1,
    "step": 2,
    "translation": "全身勾画结构",
    "creator": "Datu Medical AI Service",
    "date": datetime.now().strftime("%Y%m%d%H%M%S"),
    "description": "全身器官结构自动勾画",
    "class_dic": {str(v): k for k, v in class_dic.items()},
    "roi_list": roi_list
}
dir = r'D:\code\dipper.ai\output\release\config'
# 保存 JSON
with open(os.path.join(dir, "dipper.ai.contour.oar.fullbody.unet3d.json"), "w", encoding="utf-8") as f:
    json.dump(template, f, ensure_ascii=False, indent=2)

print("Saved json")
