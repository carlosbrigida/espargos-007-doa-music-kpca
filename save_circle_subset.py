import numpy as np
import tensorflow as tf

FILE = "espargos-0007-circle-1.tfrecords"
OUTPUT = "circle_1000.npz"

csi_list = []
pos_list = []
time_list = []

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):

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

    time = tf.io.parse_tensor(
        example.features.feature["time"].bytes_list.value[0],
        out_type=tf.float64,
    ).numpy()

    csi_list.append(csi)
    pos_list.append(pos)
    time_list.append(time)

    if (i + 1) % 100 == 0:
        print(f"Salvos {i+1} registros")

    if i >= 999:
        break

np.savez_compressed(
    OUTPUT,
    csi=np.asarray(csi_list),
    pos=np.asarray(pos_list),
    time=np.asarray(time_list),
)

print()
print("Arquivo salvo:", OUTPUT)
print("CSI shape:", np.asarray(csi_list).shape)