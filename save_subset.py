from pathlib import Path

import numpy as np
import tensorflow as tf

FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

MAX_RECORDS = 1000

csi_list = []
pos_list = []
time_list = []
rssi_list = []

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):

    if i >= MAX_RECORDS:
        break

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    pos = tf.io.parse_tensor(
        example.features.feature["pos"].bytes_list.value[0],
        out_type=tf.float64,
    ).numpy()

    t = tf.io.parse_tensor(
        example.features.feature["time"].bytes_list.value[0],
        out_type=tf.float64,
    ).numpy()

    rssi = tf.io.parse_tensor(
        example.features.feature["rssi"].bytes_list.value[0],
        out_type=tf.float32,
    ).numpy()

    csi_list.append(csi)
    pos_list.append(pos)
    time_list.append(t)
    rssi_list.append(rssi)

    if (i + 1) % 100 == 0:
        print(f"Salvos {i+1} registros")

np.savez_compressed(
    "standing_center_1000.npz",
    csi=np.asarray(csi_list),
    pos=np.asarray(pos_list),
    time=np.asarray(time_list),
    rssi=np.asarray(rssi_list),
)

print("\nArquivo salvo: standing_center_1000.npz")