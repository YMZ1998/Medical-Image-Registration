import os
import shutil
from pprint import pprint

from utils.dataset import get_test_files


def copy_test_data():
    test_files = get_test_files(os.path.join('../data', "NLST"))
    case_id = 5
    pprint(test_files[case_id])
    fixed_path = test_files[case_id]["fixed_image"]
    moving_path = test_files[case_id]["moving_image"]

    save_dir = './data'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(fixed_path, os.path.join(save_dir, "fixed.nii.gz"))
    shutil.copy(moving_path, os.path.join(save_dir, "moving.nii.gz"))
    print("Test data copied!")


if __name__ == "__main__":
    copy_test_data()
