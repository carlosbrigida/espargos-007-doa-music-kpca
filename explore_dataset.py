from pathlib import Path

import numpy as np
import tensorflow as tf

file_path = Path(
    "espargos-0007-human-helmet-standing-center-1.tfrecords"
)

positions = []
times = []

dataset = tf.data.TFRecordDataset(str(file_path))

count = 0

for raw_record in dataset:

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    pos = tf.io.parse_tensor(
        example.features.feature["pos"].bytes_list.value[0],
        out_type=tf.float64,
    ).numpy()

    t = tf.io.parse_tensor(
        example.features.feature["time"].bytes_list.value[0],
        out_type=tf.float64,
    ).numpy()

    positions.append(pos)
    times.append(t)

    count += 1

    if count % 1000 == 0:
        print(f"Lidos {count} registros...")

positions = np.asarray(positions)
times = np.asarray(times)

print("\n==============================")
print("RESUMO DO DATASET")
print("==============================")

print("\npositions shape:", positions.shape)
print("times shape:", times.shape)

print("\nPrimeira posição:")
print(positions[0])

print("\nÚltima posição:")
print(positions[-1])

print("\nTempo total [s]:")
print(times[-1] - times[0])

print("\nX range:")
print(positions[:, 0].min(), positions[:, 0].max())

print("\nY range:")
print(positions[:, 1].min(), positions[:, 1].max())

print("\nZ range:")
print(positions[:, 2].min(), positions[:, 2].max())