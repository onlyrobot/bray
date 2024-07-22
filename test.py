from bray import RemoteModel
import bray
import torch

bray.init(project="test", trial="test")
    
forward_args = (
    torch.rand(10, 10).numpy(),
)

remote_model = RemoteModel(
    name="my_model", 
    model=torch.nn.Linear(10, 10),
    forward_args=forward_args,
    use_onnx="infer",
)

print(remote_model(*forward_args))