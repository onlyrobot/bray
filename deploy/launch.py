import sys
import os
from os.path import join
import time
import shutil
import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    make_batch,
    flatten_nested_array,
)

bray_model_dir = sys.argv[1]
model_dir = f"./{os.path.basename(bray_model_dir)}"
os.makedirs(model_dir, exist_ok=True)
shutil.copy2(join(bray_model_dir, "model.onnx"), join(model_dir, "model.onnx"))

forward_inputs = torch.load(join(bray_model_dir, "forward_inputs.pt"))
batch_args, batch_kwargs = make_batch([forward_inputs])

flatten_inputs = flatten_nested_array(
    (batch_args, tuple(batch_kwargs.values())), sort_keys=True
)
ort_session = ort.InferenceSession(
    join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"]
)
inputs = dict(zip([i.name for i in ort_session.get_inputs()], flatten_inputs))

os.makedirs(join(model_dir, "forward_inputs"), exist_ok=True)
for name, input in inputs.items():
    np.save(join(model_dir, f"forward_inputs/{name}.npy"), input)

forward_outputs = torch.load(join(bray_model_dir, "forward_outputs.pt"))

os.makedirs(join(model_dir, "forward_outputs"), exist_ok=True)
outputs = flatten_nested_array(forward_outputs)
for i, output in enumerate(outputs):
    np.save(join(model_dir, f"forward_outputs/{i}.npy"), output)

# Test onnx model in python
print("-------- Test onnx model in python ----------")
ort_outputs = ort_session.run(None, inputs)
# test session run latency
beg = time.time()
for _ in range(100):
    ort_session.run(None, inputs)
print("Onnx session run latency:", (time.time() - beg) / 100 * 1000, "ms")
print("Relative and absolute error of each output:")
for i, (ort_output, output) in enumerate(zip(ort_outputs, outputs)):
    diff = np.sum(np.abs(ort_output - output))
    sum = np.sum(np.abs(output))
    cnt = np.prod(output.shape)
    print(f"{i}th output error: {diff / sum * 100 if sum > 0 else 0}%", diff / cnt)

# Test onnx model in csharp
print("-------- Test onnx model in csharp ----------")
cur_dir = os.path.dirname(__file__)
os.system(f"dotnet run --project {join(cur_dir, 'onnx/csharp/Project')} -- {model_dir}")
