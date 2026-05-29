import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FILE = "espargos-0007-circle-1.tfrecords"

positions = []
times = []

print("Lendo dataset...")

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    pos_bytes = example.features.feature["pos"].bytes_list.value[0]
    time_bytes = example.features.feature["time"].bytes_list.value[0]

    pos = tf.io.parse_tensor(
        pos_bytes,
        out_type=tf.float64,
    ).numpy()

    time = tf.io.parse_tensor(
        time_bytes,
        out_type=tf.float64,
    ).numpy()

    positions.append(pos)
    times.append(time)

    if (i + 1) % 1000 == 0:
        print(f"Lidos {i + 1} registros")

positions = np.asarray(positions)
times = np.asarray(times)

print("\n==============================")
print("RESUMO - CIRCLE")
print("==============================")

print("\nSnapshots:")
print(len(positions))

print("\nTempo total [s]:")
print(times[-1] - times[0])

print("\nX range:")
print(positions[:, 0].min(), positions[:, 0].max())

print("\nY range:")
print(positions[:, 1].min(), positions[:, 1].max())

print("\nZ range:")
print(positions[:, 2].min(), positions[:, 2].max())

plt.figure(figsize=(8, 8))

plt.plot(
    positions[:, 0],
    positions[:, 1],
    ".",
    markersize=2,
)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Trajectory - Circle Scenario")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()