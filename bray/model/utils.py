import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    make_batch,
    flatten_nested_array,
)


def build_onnx_input_names(forward_args, forward_kwargs):
    return [f"tuple_input_{i}" for i in range(len(forward_args))] + list(
        forward_kwargs.keys()
    )


def build_onnx_inputs(forward_args, forward_kwargs):
    return {
        **{f"tuple_input_{i}": input for i, input in enumerate(forward_args)},
        **forward_kwargs,
    }


def build_onnx_outputs(output_names, outputs):
    if output_names[0] != "tuple_output_0":  # outputs is a dict
        return {k: v for k, v in zip(output_names, outputs)}
    elif len(output_names) == 1:  # outputs is a single tensor
        return outputs[0]
    else:
        return tuple(outputs)


def export_onnx(
    model: torch.nn.Module,
    path: str,
    forward_args: tuple[np.ndarray] = (),
    forward_kwargs: dict[str, np.ndarray] = {},
    relative_diff: float = 1e-5,
):
    input_names = build_onnx_input_names(forward_args, forward_kwargs)
    forward_args = make_batch([forward_args])
    forward_kwargs = make_batch([forward_kwargs])

    tensor_args = handle_nested_array(forward_args, torch.from_numpy)
    tensor_kwargs = handle_nested_array(forward_kwargs, torch.from_numpy)

    with torch.no_grad():
        origin_outputs = model(*tensor_args, **tensor_kwargs)

    def validate_model_output(output):
        assert isinstance(output, torch.Tensor)
        return np.array(output)

    origin_outputs = handle_nested_array(
        origin_outputs, validate_model_output, type_check=False
    )
    if isinstance(origin_outputs, np.ndarray):
        output_names = ["tuple_output_0"]
    elif isinstance(origin_outputs, (tuple, list)):
        output_names = [f"tuple_output_{i}" for i in range(len(origin_outputs))]
    elif isinstance(origin_outputs, dict):
        output_names = list(origin_outputs.keys())
    else:
        raise ValueError("Invalid model forward output type.")
    assert len(output_names) == len(
        flatten_nested_array(origin_outputs)
    ), "Invalid model forward output, please check the output of your model."

    torch.onnx.export(
        model,
        tensor_args + (tensor_kwargs,),
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    ort_session = ort.InferenceSession(path)
    onnx_input_names = [i.name for i in ort_session.get_inputs()]
    assert input_names == onnx_input_names, "Inputs of ONNX and PyTorch are different"
    onnx_output_names = [i.name for i in ort_session.get_outputs()]
    assert (
        output_names == onnx_output_names
    ), "Outputs of ONNX and PyTorch are different"
    ort_inputs = build_onnx_inputs(forward_args, forward_kwargs)
    ort_outputs = ort_session.run(output_names, ort_inputs)

    ort_outputs = build_onnx_outputs(output_names, ort_outputs)

    assert type(origin_outputs) == type(
        ort_outputs
    ), "Outputs of ONNX and PyTorch are different"
    if isinstance(origin_outputs, dict):
        assert (
            origin_outputs.keys() == ort_outputs.keys()
        ), "Outputs of ONNX and PyTorch are different"
        assert np.allclose(
            list(origin_outputs.values()),
            list(ort_outputs.values()),
            rtol=relative_diff,
        ), "Outputs of ONNX and PyTorch are different"
    else:
        assert np.allclose(
            origin_outputs, ort_outputs, rtol=relative_diff
        ), "Outputs of ONNX and PyTorch are different"


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            # return x + y + z
            return {"xyz": x + y + z, "xy": x * y}
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
