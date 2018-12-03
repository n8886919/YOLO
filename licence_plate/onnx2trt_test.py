#! /usr/bin/python3
import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("v1/export/YOLO_export/LPD.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(1, 3, 320, 512)).astype(np.float32)
output_data = engine.run(input_data)
print(output_data)
print(output_data.shape)
