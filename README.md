# [Medical-Image-Registration](https://github.com/YMZ1998/Medical-Image-Registration)

# Dataset

NLST Task: Paired Lung CT 3D Registration with Keypoints

https://learn2reg.grand-challenge.org/Datasets/

# Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple monai
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchsummary
```

## PyInstaller Installation Guide:

### 1. Create and Activate Conda Environment

```bash
conda create --name MIR-INSTALL python=3.9
```

```bash
conda activate MIR-INSTALL
```

### 2. Install Required Python Packages

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple SimpleITK
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy
```

### 3. Use PyInstaller to Package Python Script

```bash
cd installer
pyinstaller --name MIR-LUNG --onefile --icon=mir_lung_logo.ico MIR-LUNG.py
```

### 4. Clean the Build Cache and Temporary Files

```bash
pyinstaller --clean MIR-LUNG.spec
```

### 5. Run the Executable

Once the build is complete, you can run the generated `MIR-LUNG.exe` with the required parameters:

```bash
MIR-LUNG.exe --onnx_path ./checkpoint/mir_lung.onnx --fixed_path ./data/fixed.nii.gz --moving_path ./data/moving.nii.gz --result_path ./result
```

- `--onnx_path`: Path to ONNX model.
- `--fixed_path`:Path to fixed image.
- `--moving_path`:Path to moving image.
- `--result_path`: Path where the results will be saved.

### 6. Deactivate and Remove Conda Environment

```bash
conda deactivate
conda remove --name env --all
```

# Onnx issue

https://github.com/pytorch/pytorch/issues/100790

https://github.com/Project-MONAI/MONAI/discussions/4076

# Reference

[MONAI-tutorials](https://github.com/Project-MONAI/tutorials)

[Learn2Reg](https://learn2reg.grand-challenge.org/)

[uniGradICON](https://github.com/uncbiag/uniGradICON)

[Awesome-Medical-Image-Registration](https://github.com/Alison-brie/Awesome-Medical-Image-Registration)
