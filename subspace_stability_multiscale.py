import numpy as np
import tensorflow as tf
from scipy.linalg import subspace_angles

# ==================================================
# ESCOLHA DO DATASET
# ==================================================

# FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

FILE = "espargos-0007-circle-1.tfrecords"

SIGNAL_DIM = 4
WINDOWS = [100, 250, 500, 1000]

# ==================================================
# CARREGAR CSI
# ==================================================

print("Carregando CSI...")
print("Arquivo:", FILE)

csi_list = []

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    csi_list.append(csi)

    if (i + 1) % 10000 == 0:
        print(f"Lidos {i + 1}")

csi = np.asarray(csi_list)

print("CSI shape:", csi.shape)

# ==================================================
# ANÁLISE MULTIESCALA
# ==================================================

for window in WINDOWS:
    print("\n==============================")
    print(f"WINDOW = {window} snapshots")
    print("==============================")

    subspaces = []

    for start in range(0, csi.shape[0] - window + 1, window):
        block = csi[start:start + window]

        X = block.reshape(window, -1)
        X = X.T

        R = (X @ X.conj().T) / X.shape[1]

        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]

        Us = eigvecs[:, idx[:SIGNAL_DIM]]
        subspaces.append(Us)

    max_angles = []
    mean_angles = []

    for i in range(len(subspaces) - 1):
        angles = np.degrees(
            subspace_angles(subspaces[i], subspaces[i + 1])
        )

        max_angles.append(float(angles.max()))
        mean_angles.append(float(angles.mean()))

    print("Número de janelas:", len(subspaces))
    print("Max angle médio :", np.mean(max_angles))
    print("Max angle mediano:", np.median(max_angles))
    print("Max angle máximo :", np.max(max_angles))
    print("Mean angle médio:", np.mean(mean_angles))
    print("Mean angle mediano:", np.median(mean_angles))
    print("Mean angle máximo:", np.max(mean_angles))