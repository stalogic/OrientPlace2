import reverb
import tensorflow as tf
PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"

signature = {
    'state': tf.TensorSpec(shape=[None, 852995], dtype=tf.float64),
    'orient': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'action': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'o_log_prob': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'a_log_prob': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'reward': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'next_state': tf.TensorSpec(shape=[None, 852995], dtype=tf.float64),
    'done': tf.TensorSpec(shape=[None], dtype=tf.bool)
}

table = reverb.Table(
    name="experience",
    sampler=reverb.selectors.MaxHeap(),
    remover=reverb.selectors.MinHeap(),
    max_size=10000,
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=signature
)

server = reverb.Server([table], port=12888)

dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address="localhost:12888",
    table='experience',
    max_in_flight_samples_per_worker=2
)

for sample in dataset.take(2):
    print(sample.info)
    for key, value in sample.data.items():
        print(f"{key=} shape: {value.shape} dtype: {value.dtype}, numpy.shape: {value.numpy().shape}")


# 关闭服务器
server.stop()