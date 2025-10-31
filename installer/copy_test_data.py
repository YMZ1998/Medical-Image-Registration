import os
import shutil
from pprint import pprint

from utils.dir_process import remove_and_create_dir
from utils.dataset import get_test_files


def copy_test_data(save_dir):
    test_files = get_test_files(os.path.join(r"D:\Data\MIR\NLST2023", "NLST"))
    print(len(test_files))
    case_id = 2
    pprint(test_files[case_id])
    fixed_path = test_files[case_id]["fixed_image"]
    moving_path = test_files[case_id]["moving_image"]

    remove_and_create_dir(save_dir)
    shutil.copy(fixed_path, os.path.join(save_dir, "fixed.nii.gz"))
    shutil.copy(moving_path, os.path.join(save_dir, "moving.nii.gz"))
    print("Test data copied!")


if __name__ == "__main__":
    save_dir = './data'
    save_dir2 = './dist/data'

    copy_test_data(save_dir)
    copy_test_data(save_dir2)
