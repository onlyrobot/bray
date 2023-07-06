import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    make_batch,
    flatten_nested_array,
)


def export_onnx(
    model: torch.nn.Module,
    path: str,
    forward_args: tuple[np.ndarray] = (),
    forward_kwargs: dict[str, np.ndarray] = {},
    check_consistency: bool = True,
    relative_diff: float = 1e-5,
) -> NestedArray:
    """
    将模型导出为onnx格式，并使用导出的onnx模型验证模型的输出是否正确。
    Returns:
        NestedArray: 模型的原始输出，用于恢复onnx模型输出的numpy数组的结构。
    """
    forward_args, forward_kwargs = make_batch([(forward_args, forward_kwargs)])
    tensor_args, tensor_kwargs = handle_nested_array(
        (forward_args, forward_kwargs), torch.from_numpy, sort_keys=True
    )

    torch.onnx.export(
        model,
        tensor_args + (tensor_kwargs,),
        path,
        verbose=False,
    )

    ort_session = ort.InferenceSession(path)

    input_names = [i.name for i in ort_session.get_inputs()]
    flatten_input = flatten_nested_array(
        forward_args + (forward_kwargs,), sort_keys=True
    )
    assert len(input_names) == len(flatten_input)

    ort_inputs = dict(zip(input_names, flatten_input))
    ort_outputs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        origin_outputs = model(*tensor_args, **tensor_kwargs)

    def validate_model_output(output):
        assert isinstance(output, torch.Tensor)
        return np.array(output)

    origin_outputs = handle_nested_array(
        origin_outputs, validate_model_output, type_check=False
    )

    assert len(ort_outputs) == len(origin_outputs), "Onnx model output length error."
    if not check_consistency:
        return origin_outputs
    for ort_output, origin_output in zip(
        ort_outputs, flatten_nested_array(origin_outputs)
    ):
        if np.allclose(ort_output, origin_output, rtol=relative_diff):
            continue
        print("Onnx model output is not equal to origin Torch model output.")
        print("Onnx model output:", ort_output)
        print("Origin Torch model output:", origin_output)
        print(
            "If you are sure that the model output is correct,",
            "you can set `check_consistency=False` to skip this check.",
        )
    return origin_outputs


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return x + y + z
            # return {"xyz": x + y + z, "xy": x * y}
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
