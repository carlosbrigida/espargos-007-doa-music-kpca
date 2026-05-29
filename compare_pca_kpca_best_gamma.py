import time
import numpy as np
import tensorflow as tf

from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# ==================================================
# DATASET
# ==================================================

FILE = "espargos-0007-circle-1.tfrecords"
TOTAL_RECORDS = 77076

# FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"
# TOTAL_RECORDS = 5041

# ==================================================
# CONFIGURAÇÕES
# ==================================================

MAX_SAMPLES = 1500

SIGNAL_DIM = 4

PCA_COMPONENTS = 8

KPCA_COMPONENTS = 8
KPCA_GAMMA = 3e-4

# ==================================================
# AMOSTRAGEM UNIFORME
# ==================================================

sample_indices = set(
    np.linspace(
        0,
        TOTAL_RECORDS - 1,
        MAX_SAMPLES,
        dtype=int,
    )
)

# ==================================================
# CARREGAMENTO
# ==================================================

print("Arquivo:", FILE)
print("Amostras:", MAX_SAMPLES)

t0 = time.perf_counter()

sample_csi = []

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):

    if i not in sample_indices:
        continue

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    sample_csi.append(csi)

    if len(sample_csi) % 250 == 0:
        print(f"Amostras carregadas: {len(sample_csi)}")

sample_csi = np.asarray(sample_csi)

print("\nCSI shape:", sample_csi.shape)
print(
    "Tempo leitura:",
    round(time.perf_counter() - t0, 2),
    "s",
)

# ==================================================
# MATRIZ COMPLEXA
# ==================================================

X = sample_csi.reshape(
    sample_csi.shape[0],
    -1,
)

print("Complex shape:", X.shape)

# ==================================================
# SUBESPAÇO ORIGINAL
# ==================================================

Xh = X.T

R = (Xh @ Xh.conj().T) / Xh.shape[1]

eigvals, eigvecs = np.linalg.eigh(R)

idx = np.argsort(eigvals)[::-1]

Us_original = eigvecs[:, idx[:SIGNAL_DIM]]

# ==================================================
# MATRIZ REAL
# ==================================================

X_real = np.hstack(
    [
        X.real,
        X.imag,
    ]
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_real)

n_features = X.shape[1]

# ==================================================
# FUNÇÃO AUXILIAR
# ==================================================

def evaluate_reconstruction(name, X_rec_real):

    mse = np.mean(
        (X_real - X_rec_real) ** 2
    )

    X_complex_rec = (
        X_rec_real[:, :n_features]
        + 1j * X_rec_real[:, n_features:]
    )

    Xc = X_complex_rec.T

    R_rec = (
        Xc @ Xc.conj().T
    ) / Xc.shape[1]

    eigvals_r, eigvecs_r = np.linalg.eigh(R_rec)

    idx_r = np.argsort(eigvals_r)[::-1]

    Us_rec = eigvecs_r[:, idx_r[:SIGNAL_DIM]]

    angles = np.degrees(
        subspace_angles(
            Us_original,
            Us_rec,
        )
    )

    print("\n==============================")
    print(name)
    print("==============================")

    print("MSE       :", mse)
    print("Max angle :", angles.max())
    print("Mean angle:", angles.mean())

# ==================================================
# PCA
# ==================================================

print("\nExecutando PCA...")

t1 = time.perf_counter()

pca = PCA(
    n_components=PCA_COMPONENTS,
)

Z_pca = pca.fit_transform(X_scaled)

Xrec_pca_scaled = pca.inverse_transform(
    Z_pca
)

Xrec_pca = scaler.inverse_transform(
    Xrec_pca_scaled
)

print(
    "Tempo PCA:",
    round(time.perf_counter() - t1, 2),
    "s",
)

evaluate_reconstruction(
    "PCA",
    Xrec_pca,
)

# ==================================================
# KPCA
# ==================================================

print("\nExecutando KPCA...")

t2 = time.perf_counter()

kpca = KernelPCA(
    n_components=KPCA_COMPONENTS,
    kernel="rbf",
    gamma=KPCA_GAMMA,
    fit_inverse_transform=True,
)

Z_kpca = kpca.fit_transform(
    X_scaled
)

Xrec_kpca_scaled = kpca.inverse_transform(
    Z_kpca
)

Xrec_kpca = scaler.inverse_transform(
    Xrec_kpca_scaled
)

print(
    "Tempo KPCA:",
    round(time.perf_counter() - t2, 2),
    "s",
)

evaluate_reconstruction(
    "KPCA",
    Xrec_kpca,
)

# ==================================================
# RESUMO
# ==================================================

print("\n==============================")
print("CONFIGURAÇÃO")
print("==============================")

print("PCA components :", PCA_COMPONENTS)
print("KPCA components:", KPCA_COMPONENTS)
print("KPCA gamma     :", KPCA_GAMMA)