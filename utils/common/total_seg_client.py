import requests

url = "http://localhost:8001/segment"
file_path = r"D:\debug\test.nii.gz"

with open(file_path, "rb") as f:
    files = {"file": ("test.nii.gz", f, "application/gzip")}
    response = requests.post(url, files=files)

try:
    print(response.json())
except Exception as e:
    print("无法解析 JSON:", e)
    print(response.status_code)
    print(response.text)

