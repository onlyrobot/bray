import torch
import onnxruntime as ort
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    flatten_nested_array,
)


def export_onnx(
    model: torch.nn.Module,
    path: str,
    forward_args: tuple[np.ndarray] = (),
    forward_kwargs: dict[str, np.ndarray] = {},
    export_params: bool = False,
    check_consistency: bool = True,
    relative_diff: float = 1e-5,
    quantize: bool = False,
) -> NestedArray:
    """
    将模型导出为onnx格式，并使用导出的onnx模型验证模型的输出是否正确。
    Args:
        model: 原始的torch模型，包含forward方法
        path: 导出的onnx模型的路径
        forward_args: 模型的输入的位置参数
        forward_kwargs: 模型forward的关键字参数
        export_params: 是否导出模型的参数
        check_consistency: 是否验证模型的输出是否正确
        relative_diff: 验证模型输出是否正确时的相对误差
        quantize: 是否量化模型
    Returns:
        NestedArray: 模型的原始输出，用于恢复onnx模型输出的numpy数组的结构。
    """
    tensor_args, tensor_kwargs = handle_nested_array(
        (forward_args, forward_kwargs), torch.from_numpy, sort_keys=True
    )

    with torch.no_grad():
        model.eval()
        origin_outputs = model(*tensor_args, **tensor_kwargs)

    torch.onnx.export(
        model,
        tensor_args + (tensor_kwargs,),
        path,
        # verbose=True,
        training=torch.onnx.TrainingMode.EVAL
        if export_params
        else torch.onnx.TrainingMode.TRAINING,
        # opset_version=13,
        export_params=export_params,
        do_constant_folding=False,
        # keep_initializers_as_inputs=True,
    )

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print("Quantizing model...")
        quantize_dynamic(
            model_input=path,
            model_output=path,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
        )

    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    input_names = [i.name for i in ort_session.get_inputs()]
    flatten_input = flatten_nested_array(
        forward_args + tuple(forward_kwargs.values()), sort_keys=True
    )
    if not export_params:
        # flatten_input.extend([i.detach().numpy() for i in model.parameters()])
        state_dict = model.state_dict()
        params = [
            state_dict[name].detach().numpy()
            for name in input_names[len(flatten_input) :]
        ]
        flatten_input.extend(params)
        for name in state_dict:
            if name in input_names:
                continue
            print("Warning: unused parameter:", name)
    if len(input_names) != len(flatten_input):
        print(
            f"Onnx model input length error, input_names: {input_names}, "
            + f"flatten_input length: {len(flatten_input)}",
        )
        assert False, "Onnx model input length error."

    ort_inputs = dict(zip(input_names, flatten_input))
    ort_outputs = ort_session.run(None, ort_inputs)

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
        diff = np.abs(ort_output - origin_output) / np.maximum(
            np.abs(ort_output), np.abs(origin_output)
        )
        print("Onnx model output is not equal to origin Torch model output.")
        print("Average relative error is:", np.average(diff))
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
            return x.dot(x) + y[0].dot(y[0]) + z
            # return {"xyz": x + y + z, "xy": x * y}
            return x + y + z, x * y

    model = Model()
    export_onnx(
        model,
        "model.onnx",
        (np.array(1), np.array([2, 2]), np.array([[3]])),
    )

    export_onnx(
        model,
        "model.onnx",
        (np.array(1), np.array([2, 2])),
        {"z": np.array([[3]])},
    )

    export_onnx(
        model,
        "model.onnx",
        (np.array(1),),
        {"y": np.array([2, 2]), "z": np.array([[3]])},
    )

    export_onnx(
        model,
        "model.onnx",
        (),
        {"x": np.array(1), "y": np.array([2, 2]), "z": np.array([[3]])},
    )
