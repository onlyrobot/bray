from typing import NewType, Callable, List, Dict, Tuple
import numpy as np

# 可以是单个 numpy.ndarray，也可以是 numpy.ndarray 的列表，
# 或者是 numpy.ndarray 的字典（key为str），或者是嵌套的列表/字典
NestedArray = NewType("NestedArray", any)


def handle_nested_array(inputs, handler: Callable, type_check=False, sort_keys=False):
    if isinstance(inputs, np.ndarray): return handler(inputs)
    elif isinstance(inputs, list):
        return [
            handle_nested_array(i, handler, type_check, sort_keys)
            for i in inputs
        ]
    elif isinstance(inputs, tuple):
        return tuple(
            handle_nested_array(i, handler, type_check, sort_keys)
            for i in inputs
        )
    elif isinstance(inputs, dict):
        items = inputs.items()
        items = items if not sort_keys else sorted(items)
        return {
            k: handle_nested_array(v, handler, type_check, sort_keys)
            for k, v in items
        }
    elif not type_check: return handler(inputs)
    raise TypeError(f"Unsupported type in NestedArray: {type(inputs)}")


def flatten_nested_array(inputs: NestedArray, sort_keys=False) -> List[np.ndarray]:
    flatten_arrays = []
    def flatten(array): flatten_arrays.append(array)

    handle_nested_array(inputs, flatten, sort_keys=sort_keys)
    return flatten_arrays


def unflatten_nested_array(outputs: NestedArray, flatten_arrays: List, sort_keys=False):
    flatten_arrays_iter = iter(flatten_arrays)
    next_array = lambda _: next(flatten_arrays_iter)
    return handle_nested_array(outputs, next_array, sort_keys=sort_keys)


def make_batch(
    nested_arrays: List, concate=False, parts: List[int] = None, out=None
) -> NestedArray:
    """
    Args:
        nested_arrays: 一个NestedArray列表，要求能够符合batch的要求
        concate: 是否使用np.concatenate而不是np.stack
        parts: 如果是concate，这里输出了每个数组的维度，用于拆分
    """
    flatten_arrays = [flatten_nested_array(a) for a in nested_arrays]
    arrays = list(zip(*flatten_arrays))
    outs = flatten_nested_array(out) if out else [None] * len(arrays)
    try:
        batch = np.concatenate if concate else np.stack
        batch_arrays = [
            batch(a, out=o) for a, o in zip(arrays, outs)]
    except ValueError:
        print("Error: the arrays have different shapes or dtypes")
        print("Batch signatures: ")
        for signature in handle_nested_array(nested_arrays, 
            lambda x: (x.shape, x.dtype) if isinstance(x, np.ndarray) else x
        ): print(signature)
        raise
    if parts is not None:
        parts.extend([len(arrays[0]) for arrays in flatten_arrays])
    return unflatten_nested_array(
        nested_arrays[0] if nested_arrays else [], batch_arrays)


def split_batch(batch: NestedArray, parts: List[int] = None) -> List:
    flatten_arrays = flatten_nested_array(batch)
    if isinstance(parts, int):
        flatten_arrays = [np.array_split(a, parts) for a in flatten_arrays]
    if isinstance(parts, list):
        indices = np.cumsum(parts)
        flatten_arrays = [np.split(a, indices[:-1]) for a in flatten_arrays]
    arrays = zip(*flatten_arrays)
    return [unflatten_nested_array(batch, a) for a in arrays]


class ArrayMeta:
    def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype


def serialize_nested_array(nested_array) -> Tuple[NestedArray, bytes]:
    flatten_arrays = flatten_nested_array(nested_array)
    signature = handle_nested_array(
        nested_array, lambda x: ArrayMeta(x.shape, x.dtype))
    # array = np.concatenate([a.flatten().view(np.int8) for a in flatten_arrays])
    return signature, b"".join([a.data for a in flatten_arrays])


def deserialize_nested_array(signature, data: bytes) -> NestedArray:
    offset, array = 0, np.frombuffer(data, dtype=np.int8)

    def deserialize_array(array_meta: ArrayMeta):
        nonlocal offset
        shape, dtype = array_meta.shape, array_meta.dtype
        size = np.prod(shape) * dtype.itemsize
        offset += size
        return array[offset - size: offset].view(dtype).reshape(shape)
    return handle_nested_array(signature, deserialize_array)


if __name__ == "__main__":
    # test flatten_nested_array and unflatten_nested_array
    nested_array = [
        np.array([1, 2, 3]), np.array([4, 5, 6]),
        {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
    ]
    flatten_arrays = flatten_nested_array(nested_array)
    restored_nested_array = unflatten_nested_array(nested_array, flatten_arrays)
    assert np.allclose(nested_array[0], restored_nested_array[0])
    print("test flatten_nested_array and unflatten_nested_array passed")
    # test make_batch
    nested_arrays = [
    [
        np.array([[1, 2, 3]]), np.array([4, 5, 6]),
        {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
    ],
    [
        np.array([[1, 2, 3]]), np.array([4, 5, 6]),
        {"a": np.array([7, 8, 9]), "b": np.array([10, 11, 12])},
    ],
    ]
    batch = make_batch(nested_arrays)
    assert np.allclose(batch[0], np.vstack([nested_arrays[0][0], nested_arrays[1][0]]))
    split_array = split_batch(batch)
    np.testing.assert_array_equal(split_array[0][0], nested_arrays[0][0])
    batch2 = make_batch(nested_arrays, concate=True)
    batch3 = make_batch(nested_arrays, concate=True, out=batch2)
    split_array2 = split_batch(batch2)
    np.testing.assert_array_equal(split_array2[0][0], nested_arrays[0][0][0])
    np.testing.assert_array_equal(split_array2[0][1], nested_arrays[0][1][0])
    nested_arrays = [
    [
        np.array([[1, 2, 3]]), np.array([[4]]),
        {"a": np.array([[7, 8, 9]]), "b": np.array([[10, 11, 12]])},
    ],
    [
        np.array([[1, 2, 3], [3, 2, 1]]), np.array([[4], [5]]),
        {
            "a": np.array([[7, 8, 9], [9, 8, 7]]),
            "b": np.array([[10, 11, 12], [12, 11, 10]]),
        },
    ],
    ]
    parts = []
    batch = make_batch(nested_arrays, concate=True, parts=parts)
    assert parts == [1, 2]
    split_array = split_batch(batch, parts=parts)
    np.testing.assert_array_equal(split_array[0][0], nested_arrays[0][0])
    np.testing.assert_array_equal(split_array[0][1], nested_arrays[0][1])
    np.testing.assert_array_equal(split_array[1][0], nested_arrays[1][0])
    np.testing.assert_array_equal(split_array[1][1], nested_arrays[1][1])
    print("test make_batch passed")

    serialize_arrays = deserialize_nested_array(
        *serialize_nested_array(nested_arrays))
    np.testing.assert_array_equal(serialize_arrays[0][0], nested_arrays[0][0])
    np.testing.assert_array_equal(serialize_arrays[0][1], nested_arrays[0][1])
    np.testing.assert_array_equal(serialize_arrays[1][0], nested_arrays[1][0])
    np.testing.assert_array_equal(serialize_arrays[1][1], nested_arrays[1][1])
    print("test serialize_nested_array and deserialize_nested_array passed")
