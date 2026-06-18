import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


def wavelength_from_frequency(frequency_hz: float) -> float:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    return SPEED_OF_LIGHT / frequency_hz


def steering_vector_ula(
    angle_rad: float,
    n_antennas: int,
    antenna_spacing_m: float,
    frequency_hz: float,
) -> np.ndarray:
    wavelength = wavelength_from_frequency(frequency_hz)

    antenna_indices = np.arange(n_antennas)

    phase = (
        -2.0j
        * np.pi
        * antenna_spacing_m
        * antenna_indices
        * np.sin(angle_rad)
        / wavelength
    )

    return np.exp(phase)[:, None]


def steering_vector_azimuth_3d(
    angle_deg: float,
    antenna_positions: np.ndarray,
    array_center: np.ndarray,
    array_right: np.ndarray,
    array_normal: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    wavelength = wavelength_from_frequency(frequency_hz)

    theta = np.deg2rad(angle_deg)

    right = array_right / np.linalg.norm(array_right)
    normal = array_normal / np.linalg.norm(array_normal)

    direction = np.cos(theta) * normal + np.sin(theta) * right

    relative_positions = antenna_positions - array_center

    phase = -2.0j * np.pi / wavelength * (
        relative_positions @ direction
    )

    return np.exp(phase)[:, None]