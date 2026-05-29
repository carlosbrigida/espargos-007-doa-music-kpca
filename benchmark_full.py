import time
import numpy as np
import tensorflow as tf

# ==================================================
# ESCOLHA DO DATASET
# ==================================================

# FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

FILE = "espargos-0007-circle-1.tfrecords"

# ==================================================
# LEITURA
# ==================================================

t0 = time.perf_counter()

csi_list = []

print("Carregando CSI completo...")
print("Arquivo:", FILE)

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

t_load = time.perf_counter()

# ==================================================
# ARRAY CSI
# ==================================================

csi = np.asarray(csi_list)

print("\nCSI shape:", csi.shape)
print("CSI size [GB]:", csi.nbytes / (1024 ** 3))

# ==================================================
# MATRIZ DE DADOS
# ==================================================

X = csi.reshape(csi.shape[0], -1)

print("X shape:", X.shape)

t_reshape = time.perf_counter()

# ==================================================
# COVARIÂNCIA
# ==================================================

print("\nCalculando covariância...")

Xh = X.T

R = (Xh @ Xh.conj().T) / Xh.shape[1]

t_cov = time.perf_counter()

# ==================================================
# AUTOVALORES
# ==================================================

print("Calculando autovalores...")

eigvals = np.linalg.eigvalsh(R)
eigvals = np.sort(eigvals)[::-1]

t_eig = time.perf_counter()

# ==================================================
# ENERGIA ACUMULADA
# ==================================================

energy = eigvals / eigvals.sum()
cum_energy = np.cumsum(energy)

# ==================================================
# BENCHMARK
# ==================================================

print("\n==============================")
print("BENCHMARK")
print("==============================")

print(f"Leitura TFRecord : {t_load - t0:.2f} s")
print(f"Reshape          : {t_reshape - t_load:.2f} s")
print(f"Covariância      : {t_cov - t_reshape:.2f} s")
print(f"Autovalores      : {t_eig - t_cov:.2f} s")
print(f"Tempo total      : {t_eig - t0:.2f} s")

# ==================================================
# RESULTADOS
# ==================================================

print("\n==============================")
print("RESULTADOS")
print("==============================")

print("Snapshots:", csi.shape[0])
print("Features complexas:", X.shape[1])

print("\nTop 10 eigenvalues:\n")

for i in range(10):
    print(f"{i+1:2d}: {eigvals[i]:.6e}")

print("\nEigengap λ4/λ5:")

print(eigvals[3] / eigvals[4])

print("\nComponentes necessários:")

for threshold in [0.90, 0.95, 0.99]:
    k = np.searchsorted(cum_energy, threshold) + 1
    print(f"{threshold*100:.0f}% -> {k}")