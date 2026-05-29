import time
import numpy as np
import tensorflow as tf
from scipy.linalg import subspace_angles
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# ==================================================
# ESCOLHA DO DATASET
# ==================================================

# FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"
# TOTAL_RECORDS = 5041

FILE = "espargos-0007-circle-1.tfrecords"
TOTAL_RECORDS = 77076

# ==================================================
# CONFIGURAÇÕES
# ==================================================

SIGNAL_DIM = 4
MAX_SAMPLES = 1500
N_COMPONENTS = 8

GAMMAS = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

sample_indices = set(
    np.linspace(0, TOTAL_RECORDS - 1, MAX_SAMPLES, dtype=int)
)

print("Arquivo:", FILE, flush=True)
print("Amostras:", MAX_SAMPLES, flush=True)

# ==================================================
# CARREGAR AMOSTRA UNIFORME
# ==================================================

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
        print(f"Amostras carregadas: {len(sample_csi)}", flush=True)

sample_csi = np.asarray(sample_csi)

print("\nSample CSI:", sample_csi.shape, flush=True)
print(f"Tempo leitura: {time.perf_counter() - t0:.2f} s", flush=True)

# ==================================================
# SUBESPAÇO ORIGINAL DA AMOSTRA
# ==================================================

X = sample_csi.reshape(sample_csi.shape[0], -1)

Xh = X.T
R = (Xh @ Xh.conj().T) / Xh.shape[1]

eigvals, eigvecs = np.linalg.eigh(R)
idx = np.argsort(eigvals)[::-1]
Us_original = eigvecs[:, idx[:SIGNAL_DIM]]

# ==================================================
# DADOS REAIS PADRONIZADOS
# ==================================================

X_real = np.hstack([X.real, X.imag])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_real)

n_complex_features = X.shape[1]

# ==================================================
# KPCA GAMMA SWEEP
# ==================================================

print("\n====================================================")
print("KPCA GAMMA SWEEP")
print("====================================================")
print(
    f"{'gamma':>10} | {'MSE':>12} | {'max angle':>12} | {'mean angle':>12}"
)
print("-" * 60, flush=True)

best = None

for gamma in GAMMAS:
    t_gamma = time.perf_counter()

    kpca = KernelPCA(
        n_components=N_COMPONENTS,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
    )

    Z = kpca.fit_transform(X_scaled)
    X_rec_scaled = kpca.inverse_transform(Z)

    X_rec_real = scaler.inverse_transform(X_rec_scaled)

    mse = np.mean((X_real - X_rec_real) ** 2)

    X_rec_complex = (
        X_rec_real[:, :n_complex_features]
        + 1j * X_rec_real[:, n_complex_features:]
    )

    Xc = X_rec_complex.T
    R_rec = (Xc @ Xc.conj().T) / Xc.shape[1]

    eigvals_r, eigvecs_r = np.linalg.eigh(R_rec)
    idx_r = np.argsort(eigvals_r)[::-1]
    Us_rec = eigvecs_r[:, idx_r[:SIGNAL_DIM]]

    angles = np.degrees(subspace_angles(Us_original, Us_rec))

    result = {
        "gamma": gamma,
        "mse": float(mse),
        "max_angle": float(angles.max()),
        "mean_angle": float(angles.mean()),
        "time": time.perf_counter() - t_gamma,
    }

    print(
        f"{gamma:10.1e} | "
        f"{result['mse']:12.3f} | "
        f"{result['max_angle']:12.6f} | "
        f"{result['mean_angle']:12.6f} | "
        f"time={result['time']:.2f}s",
        flush=True,
    )

    if best is None or result["mean_angle"] < best["mean_angle"]:
        best = result

print("\n====================================================")
print("MELHOR GAMMA")
print("====================================================")
print("gamma:", best["gamma"])
print("MSE:", best["mse"])
print("Max angle:", best["max_angle"])
print("Mean angle:", best["mean_angle"])
print("\nTempo total:", time.perf_counter() - t0, "s")