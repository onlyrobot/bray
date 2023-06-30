from typing import NewType
import numpy as np

# 可以是单个 numpy.ndarray，也可以是 numpy.ndarray 的列表，
# 或者是 numpy.ndarray 的字典（key为str），或者是嵌套的列表/字典
NestedArray = NewType("NestedArray", any)


def handle_nested_array(inputs, handler: callable, type_check=True):
    if isinstance(inputs, np.ndarray):
        return handler(inputs)
    elif isinstance(inputs, list):
        return [handle_nested_array(i, handler, type_check) for i in inputs]
    elif isinstance(inputs, tuple):
        return tuple(handle_nested_array(i, handler, type_check) for i in inputs)
    elif isinstance(inputs, dict):
        # sorted_items = sorted(list(inputs.items()))
        sorted_items = inputs.items()
        return {
            k: handle_nested_array(
                v,
                handler,
                type_check,
            )
            for k, v in sorted_items
        }
    elif type_check:
        raise TypeError(f"Unsupported type in NestedArray: {type(inputs)}")
    else:
        return handler(inputs)


def flatten_nested_array(inputs: NestedArray) -> list[NestedArray]:
    flatten_arrays = []

    def flatten(array):
        flatten_arrays.append(array)

    handle_nested_array(inputs, flatten)
    return flatten_arrays


def unflatten_nested_array(
    inputs: NestedArray, flatten_arrays: list[NestedArray], index=-1
) -> NestedArray:
    def unflatten(_):
        nonlocal index
        index += 1
        return flatten_arrays[index]

    return handle_nested_array(inputs, unflatten)


def make_batch(nested_arrays: list[NestedArray]) -> NestedArray:
    if len(nested_arrays) == 0:
        return []
    flatten_arrays = [flatten_nested_array(arrays) for arrays in nested_arrays]
    arrays = zip(*flatten_arrays)
    batch_arrays = [np.stack(a) for a in arrays]
    return unflatten_nested_array(nested_arrays[0], batch_arrays)


if __name__ == "__main__":
    # test flatten_nested_array and unflatten_nested_array
    nested_array = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
    ]
    flatten_arrays = flatten_nested_array(nested_array)
    restored_nested_array = unflatten_nested_array(nested_array, flatten_arrays)
    assert np.allclose(nested_array[0], restored_nested_array[0])
    print("test flatten_nested_array and unflatten_nested_array passed")
    # test make_batch
    nested_arrays = [
        [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
        ],
        [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
        ],
    ]
    batch = make_batch(nested_arrays)
    assert np.allclose(batch[0], np.vstack([nested_arrays[0][0], nested_arrays[1][0]]))
    print("test make_batch passed")
