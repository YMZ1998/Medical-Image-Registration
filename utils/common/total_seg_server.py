import shutil
import uuid
from asyncio import Lock

from fastapi import FastAPI, UploadFile, File
from totalsegmentator.config import *
from totalsegmentator.python_api import totalsegmentator

print(setup_totalseg())  # 仅执行一次，服务器启动时初始化
print(get_weights_dir(), get_totalseg_dir())

app = FastAPI()

UPLOAD_FOLDER = r"D:\debug\seg_uploads"
OUTPUT_FOLDER = r"D:\\debug\\seg_results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

lock = Lock()


# uvicorn utils.common.total_seg_server:app --host 0.0.0.0 --port 8000 --reload


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    async with lock:
        case_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{case_id}.nii.gz")
        out_dir = os.path.join(OUTPUT_FOLDER, case_id)
        print(input_path)

        try:
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            os.makedirs(out_dir, exist_ok=True)

            totalsegmentator(input_path, out_dir, fast=True,
                             device="gpu" if torch.cuda.is_available() else "cpu",
                             verbose=True)

            return {"case_id": case_id, "output_dir": out_dir}

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            return {"error": str(e), "traceback": tb}


# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# curl -X POST "http://localhost:8000/segment" -F "file=@example.nii.gz"
