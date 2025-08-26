import os
import sys
import shutil
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

file_name = "dtcp.image.scene.dll"
src = os.path.join(r"D:\Code\install\dtcp.common\bin\release", file_name)
dst = os.path.join(r"C:\Program Files (x86)\rtStation", file_name)

if is_admin():
    # 已经是管理员，直接执行复制
    shutil.copyfile(src, dst)
    print("复制成功！")
else:
    # 重新以管理员权限运行
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1
    )
