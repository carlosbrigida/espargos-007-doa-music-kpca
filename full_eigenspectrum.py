import numpy as np
import tensorflow as tf

# ==================================================
# ESCOLHA O DATASET
# ==================================================

#FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

FILE = "espargos-0007-circle-1.tfrecords"

# ==================================================
# LEITURA CSI
# ==================================================

csi_list = []

print("Carregando CSI...")

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    csi_list.append(csi)

    if (i + 1) % 5000 == 0:
        print(f"Lidos {i + 1}")

csi = np.asarray(csi_list)

print("\nCSI shape:", csi.shape)

# ==================================================
# MATRIZ DE DADOS
# ==================================================

X = csi.reshape(csi.shape[0], -1)

print("X shape:", X.shape)

# ==================================================
# COVARIÂNCIA
# ==================================================

print("\nCalculando covariância...")

Xh = X.T

R = (Xh @ Xh.conj().T) / Xh.shape[1]

# ==================================================
# AUTOVALORES
# ==================================================

print("Calculando autovalores...")

eigvals = np.linalg.eigvalsh(R)
eigvals = np.sort(eigvals)[::-1]

energy = eigvals / eigvals.sum()
cum_energy = np.cumsum(energy)

# ==================================================
# RESULTADOS
# ==================================================

print("\n==============================")
print("RESULTADOS")
print("==============================")

print("\nTop 20 eigenvalues:\n")

for i in range(20):
    print(f"{i+1:2d}: {eigvals[i]:.6e}")

print("\nComponents needed:")

for threshold in [0.90, 0.95, 0.99]:
    k = np.searchsorted(cum_energy, threshold) + 1
    print(f"{threshold*100:.0f}% -> {k}")

print("\nEigengap λ4/λ5:")

print(eigvals[3] / eigvals[4])