import os
import shutil


def copy_checkpoint():
    src = '../results/seg_resnet/model_with_warp.onnx'
    dst = './checkpoint/mir_lung.onnx'
    os.makedirs(dst, exist_ok=True)
    shutil.copy(src, dst)


if __name__ == '__main__':
    copy_checkpoint()
