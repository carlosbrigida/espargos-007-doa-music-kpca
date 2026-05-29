from pathlib import Path

import numpy as np
import tensorflow as tf

FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

all_csi = []

dataset = tf.data.TFRecordDataset(FILE)

count = 0

for raw_record in dataset:

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    all_csi.append(csi)

    count += 1

    if count % 1000 == 0:
        print(f"Lidos {count} snapshots")

all_csi = np.asarray(all_csi)

print("\n==============================")
print("CSI")
print("==============================")

print("Shape:", all_csi.shape)
print("Dtype:", all_csi.dtype)

mag = np.abs(all_csi)

print("\nMagnitude")

print("Mean :", mag.mean())
print("Std  :", mag.std())
print("Min  :", mag.min())
print("Max  :", mag.max())

phase = np.angle(all_csi)

print("\nPhase")

print("Mean :", phase.mean())
print("Std  :", phase.std())
print("Min  :", phase.min())
print("Max  :", phase.max())