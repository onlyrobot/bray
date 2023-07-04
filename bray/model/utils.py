import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    make_batch,
)


def export_onnx(
    model: torch.nn.Module,
    path: str,
    forward_args: tuple[np.ndarray] = (),
    forward_kwargs: dict[str, np.ndarray] = {},
):
    input_names = [f"input_{i}" for i in range(len(forward_args))] + list(
        forward_kwargs.keys()
    )
    forward_args = make_batch([forward_args])
    forward_kwargs = make_batch([forward_kwargs])

    tensor_args = handle_nested_array(forward_args, torch.from_numpy)
    tensor_kwargs = handle_nested_array(forward_kwargs, torch.from_numpy)

    torch.onnx.export(
        model,
        tensor_args + (tensor_kwargs,),
        path,
        verbose=True,
        input_names=input_names,
    )
    ort_session = ort.InferenceSession(path)
    ort_inputs = {
        **{f"input_{i}": input for i, input in enumerate(forward_args)},
        **forward_kwargs,
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        origin_outputs = model(*tensor_args, **tensor_kwargs)
    origin_outputs = handle_nested_array(origin_outputs, np.array, type_check=False)

    if not isinstance(origin_outputs, tuple):
        origin_outputs = [origin_outputs]
    assert len(origin_outputs) == len(ort_outputs)
    for origin_output, ort_output in zip(origin_outputs, ort_outputs):
        assert np.allclose(
            origin_output, ort_output
        ), "Outputs of ONNX and PyTorch are different"


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return x + y + z, x * y

    model = Model()
    export_onnx(
        model,
        "model.onnx",
        (np.array(1), np.array(2), np.array(3)),
    )

    export_onnx(
        model,
        "model.onnx",
        (np.array(1), np.array(2)),
        {"z": np.array(3)},
    )

    export_onnx(
        model,
        "model.onnx",
        (np.array(1),),
        {"y": np.array(2), "z": np.array(3)},
    )

    export_onnx(
        model,
        "model.onnx",
        (),
        {"x": np.array(1), "y": np.array(2), "z": np.array(3)},
    )
