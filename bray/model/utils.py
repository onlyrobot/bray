import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    flatten_nested_array,
    make_batch,
)


def export_onnx(
    model: torch.nn.Module,
    forward_args: tuple[NestedArray],
    path: str,
):
    inputs = make_batch([forward_args])
    flattened_inputs = flatten_nested_array(inputs)
    input_names = [f"input_{i}" for i in range(len(flattened_inputs))]

    inputs = tuple(handle_nested_array(inputs, torch.from_numpy))

    torch.onnx.export(
        model,
        inputs,
        path,
        verbose=True,
        input_names=input_names,
    )
    ort_session = ort.InferenceSession(path)
    outputs = ort_session.run(
        None,
        {f"input_{i}": flattened_inputs[i] for i in range(len(flattened_inputs))},
    )

    origin_outputs = model(*forward_args)
    origin_outputs = handle_nested_array(origin_outputs, np.array, type_check=False)
    origin_outputs = flatten_nested_array(origin_outputs)

    assert len(origin_outputs) == len(outputs)
    for origin_output, output in zip(origin_outputs, outputs):
        assert np.allclose(
            origin_output, output
        ), "Outputs of ONNX and PyTorch are different"


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x["a"] + y

    model = Model()
    export_onnx(
        model,
        ({"a": np.array(1)}, np.array(2)),
        "model.onnx",
    )
