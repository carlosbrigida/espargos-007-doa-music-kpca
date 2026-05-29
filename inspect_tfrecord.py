from pathlib import Path
import tensorflow as tf
import numpy as np

file_path = Path("espargos-0007-human-helmet-standing-center-1.tfrecords")

dataset = tf.data.TFRecordDataset(str(file_path))

for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi_bytes = example.features.feature["csi"].bytes_list.value[0]
    rssi_bytes = example.features.feature["rssi"].bytes_list.value[0]
    pos_bytes = example.features.feature["pos"].bytes_list.value[0]
    time_bytes = example.features.feature["time"].bytes_list.value[0]

    csi = tf.io.parse_tensor(csi_bytes, out_type=tf.complex64).numpy()
    rssi = tf.io.parse_tensor(rssi_bytes, out_type=tf.float32).numpy()
    pos = tf.io.parse_tensor(pos_bytes, out_type=tf.float64).numpy()
    time = tf.io.parse_tensor(time_bytes, out_type=tf.float64).numpy()

    print("CSI")
    print("shape:", csi.shape)
    print("dtype:", csi.dtype)
    print("primeiro valor:", csi.flatten()[0])
    print()

    print("RSSI")
    print("shape:", rssi.shape)
    print("dtype:", rssi.dtype)
    print(rssi)
    print()

    print("POS")
    print("shape:", pos.shape)
    print("dtype:", pos.dtype)
    print(pos)
    print()

    print("TIME")
    print("shape:", time.shape)
    print("dtype:", time.dtype)
    print(time)