@echo off

MIR-LUNG.exe --onnx_path ./checkpoint/mir_lung.onnx --fixed_path ./data/fixed.nii.gz --moving_path ./data/moving.nii.gz --result_path ./result

pause
