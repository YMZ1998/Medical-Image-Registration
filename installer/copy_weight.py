import os
import shutil

from parse_args import parse_args
from utils.dir_process import remove_and_create_dir


def copy_checkpoint(src, dst):
    # remove_and_create_dir(dst)
    print('Copy checkpoint from {} to {}'.format(src, dst))
    shutil.copy(src, dst)


if __name__ == '__main__':
    args = parse_args()
    arch=args.arch
    print('Arch: {}'.format(arch))
    src = f'../results/{arch}'
    dst = './checkpoint'
    dst2 = './dist/checkpoint'
    copy_checkpoint(os.path.join(src, 'model.onnx'), os.path.join(dst, 'mir_lung.onnx'))
    copy_checkpoint(os.path.join(src, 'model_with_warp.onnx'), os.path.join(dst, 'mir_lung2.onnx'))
