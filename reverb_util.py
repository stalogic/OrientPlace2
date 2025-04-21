import time
import torch
import numpy as np
import tensorflow as tf
from loguru import logger
from collections import defaultdict

TORCH2TF_DTYPE = {
    torch.float32: tf.float32,
    torch.float64: tf.float64,
    torch.int32: tf.int32,
    torch.int64: tf.int64,
    torch.bool: tf.bool
}

def to_tf_dtype(torch_dtype):
    return TORCH2TF_DTYPE[torch_dtype]

TF2TORCH_DTYPE = {
    tf.float32: torch.float32,
    tf.float64: torch.float64,
    tf.int32: torch.int32,
    tf.int64: torch.int64,
    tf.bool: torch.bool
}

def tf_to_torch(tf_tensor):
    """将 TensorFlow 张量转换为 PyTorch 张量"""
    numpy_array = tf_tensor.numpy()
    torch_dtype = TF2TORCH_DTYPE.get(tf_tensor.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported TensorFlow dtype: {tf_tensor.dtype}")
    return torch.tensor(numpy_array, dtype=torch_dtype)

def torch_to_tf(torch_tensor):
    """将 PyTorch 张量转换为 TensorFlow 张量"""
    numpy_array = torch_tensor.cpu().numpy()
    tf_dtype = TORCH2TF_DTYPE.get(torch_tensor.dtype)
    if tf_dtype is None:
        raise ValueError(f"Unsupported PyTorch dtype: {torch_tensor.dtype}")
    return tf.convert_to_tensor(numpy_array, dtype=tf_dtype)

class PlaceTrajectoryDataset:
    def __init__(self, dataset: tf.data.Dataset, batch_size: int):
        self.dataset = dataset
        self.iterator = iter(dataset)
        self.batch_size = batch_size
        self.model_id = None
        self.generator = None

    def read(self, model_id=None):
        if model_id is not None:
            self.model_id = model_id
        if self.generator is None:
            self.generator = self.__iter__()
        return next(self.generator)

    def __iter__(self):
        def generator():
            batches = defaultdict(list)
            discarded = 0
            first = True
            while True:
                t0 = time.time()
                batch = next(self.iterator).data
                t1 = time.time()
                if self.model_id is not None:
                    model_ids = batch.pop('model_id')
                    valid_mask = model_ids.numpy() == self.model_id
                    discarded += np.sum(~valid_mask)
                    batch = tf.nest.map_structure(lambda x: x[valid_mask].numpy(), batch)
                else:
                    batch = tf.nest.map_structure(lambda x: x.numpy(), batch)
                t2 = time.time()

                for i in range(len(batch['action'])):
                    for key, value in batch.items():
                        batches[key].append(value[i])

                    if len(batches['action']) >= self.batch_size:
                        t3 = time.time()
                        result = {}
                        for key, value in batches.items():
                            array = np.concatenate(value, axis=0)
                            if len(array.shape) == 1:
                                array = np.expand_dims(array, axis=1)
                            # elif len(array.shape) > 2:
                            #     raise ValueError(f"Invalid shape: {array.shape} for `{key}`")
                            result[key] = array

                        if discarded > 0:
                            logger.info(f"Collect {self.batch_size} samples, discard {discarded} samples")
                        if first:
                            logger.info(", ".join([f"{key}: ({value.shape}, {value.dtype})" for key, value in result.items()]))
                            first = False
                        logger.info(f"Read {self.batch_size} samples, batch: {t1 - t0:.2f}s, tf2np: {t2 - t1:.2f}s, batches: {t3 - t2:.2f}s, array: {time.time() - t3:.2f}s")
                        yield result
                        batches.clear()
                        discarded = 0

        return generator()
