import os
import shutil

from utils.dir_process import remove_and_create_dir


def copy_checkpoint(src, dst):
    # remove_and_create_dir(dst)
    shutil.copy(src, os.path.join(dst, 'mir_lung.onnx'))


if __name__ == '__main__':
    src = '../results/seg_resnet/model.onnx'
    src2 = '../results/seg_resnet/model_with_warp.onnx'
    dst = './checkpoint'
    dst2 = './dist/checkpoint'
    copy_checkpoint(src, dst)
    copy_checkpoint(src, dst2)
