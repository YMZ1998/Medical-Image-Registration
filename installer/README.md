                                                                                                                                                                                                        ## PyInstaller Installation Guide:

### 1. Use PyInstaller to Package Python Script

```bash
cd installer
pyinstaller --name MIR-LUNG --onefile --icon=mir_lung_logo.ico MIR-LUNG.py
```

### 2. Clean the Build Cache and Temporary Files

```bash
pyinstaller --clean MIR-LUNG.spec
```

### 3. Run the Executable

Once the build is complete, you can run the generated `MIR-LUNG.exe` with the required parameters:

```bash
MIR-LUNG.exe --fixed_path ./data/fixed.nii.gz --moving_path  ./data/fixed.nii.gz --result_path ./result