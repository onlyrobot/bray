from typing import Iterator, Callable
import torch
from threading import Thread, Condition

from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)
import numpy as np
from bray.metric.metric import merge


def handle(replay: list):
    pass

class ImpalaBatchBuffer:
    def __init__(self, buffer: Iterator[NestedArray], batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __next__(self) -> NestedArray:
        now_batch_size = 0
        exps_list = []
        seq_len_plus_1 = []
        while now_batch_size < self.batch_size:
            exps = next(self.buffer)
            exp_cnt = len(exps)
            seq_len_plus_1.append(exp_cnt)
            exps_list.extend(exps)
            now_batch_size += exp_cnt
        return self._pad_and_reshape(exps_list, seq_len_plus_1)

    def __iter__(self) -> Iterator[NestedArray]:
        return self

    def _pad_and_reshape(self, exp_list, seq_len_plus_1):
        training_data = {}
        training_obs = [exp['obs'] for exp in exp_list]
        num_head = len(training_obs[0])
        training_data['obs'] = [np.array([obs[i] for obs in training_obs], dtype=np.float32) for i in range(num_head)]
        training_data['action'] = np.array([exp['action'] for exp in exp_list], dtype=np.float32)
        training_data['rewards'] = np.array([exp['rewards'] for exp in exp_list], dtype=np.float32)
        training_data['old_log_probs'] = np.array([exp['old_log_probs'] for exp in exp_list], dtype=np.float32)
        training_data['discounts'] = np.array([exp['discounts'] for exp in exp_list], dtype=np.float32)

        attr_names = ['rewards', 'old_log_probs', 'discounts']
        seq_len_plus_1 = np.array(seq_len_plus_1, dtype=np.int32)
        valid_seq_len = seq_len_plus_1 - 1
        valid_seq_idx = []
        tail_seq_idx = np.cumsum(seq_len_plus_1) - 1
        for _id in range(tail_seq_idx[-1]):
            if _id in tail_seq_idx:
                continue
            valid_seq_idx.append(_id)
        valid_seq_idx = np.array(valid_seq_idx)
        training_data["seq_len_plus_1"] = seq_len_plus_1
        training_data["tail_seq_idx"] = tail_seq_idx
        training_data["valid_seq_idx"] = valid_seq_idx
        max_seq_len = np.max(valid_seq_len)
        begin_seq_idx = np.concatenate([[0], np.cumsum(valid_seq_len)[:-1]], axis=0)
        slice_idx = np.stack([begin_seq_idx, valid_seq_len], axis=1)
        for attr_name in attr_names:
            raw_array = training_data[attr_name]
            valid_raw_array = np.take(raw_array, valid_seq_idx)
            slice_array = []
            for x in slice_idx:
                slice_array.append(np.pad(valid_raw_array[x[0]:(x[0] + x[1])], (max_seq_len - x[1], 0),
                                          'constant', constant_values=(0, 0)))
            slice_array = np.transpose(np.array(slice_array))
            training_data[attr_name] = slice_array
        return training_data
