import shutil

if __name__ == '__main__':
    src = './dist/MIR-LUNG.exe'
    dst = r'D:\code\dipper.ai\resource\ai_reg\MIR-LUNG.exe'
    shutil.copy(src, dst)
