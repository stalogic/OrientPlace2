import torch
import tensorflow as tf

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