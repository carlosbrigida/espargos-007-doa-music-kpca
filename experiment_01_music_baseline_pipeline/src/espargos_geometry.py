from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReceiverArray:
    name: str
    center: np.ndarray
    right: np.ndarray
    up: np.ndarray
    spacing_horizontal: float = 0.06
    spacing_vertical: float = 0.06
    n_rows: int = 2
    n_cols: int = 4


@dataclass(frozen=True)
class Transmitter:
    name: str
    position: np.ndarray
    mac: str


RECEIVER_ARRAYS = {
    0: ReceiverArray(
        name="espargosnorth",
        center=np.array([-0.0090, 6.7896, -1.0058]),
        right=np.array([0.9892, 0.1464, -0.0039]),
        up=np.array([0.0571, -0.3641, 0.9296]),
    ),
    1: ReceiverArray(
        name="espargoswest",
        center=np.array([-2.5101, 3.6108, -1.0331]),
        right=np.array([-0.0472, 0.9989, 0.0075]),
        up=np.array([0.2173, 0.0018, 0.9761]),
    ),
    2: ReceiverArray(
        name="espargossouth",
        center=np.array([-0.9266, 0.4876, -1.0233]),
        right=np.array([-0.9509, 0.3086, -0.0219]),
        up=np.array([0.0968, 0.3639, 0.9264]),
    ),
    3: ReceiverArray(
        name="espargoseast",
        center=np.array([4.4188, 2.8471, -1.0134]),
        right=np.array([-0.1077, -0.9942, 0.0008]),
        up=np.array([-0.1797, 0.0213, 0.9835]),
    ),
}


TRANSMITTERS = {
    "tx1": Transmitter(
        name="TX1",
        position=np.array([0.2708, 6.4101, 0.5188]),
        mac="0a:ee:f5:00:00:01",
    ),
    "tx2": Transmitter(
        name="TX2",
        position=np.array([3.9133, 4.1573, 0.5219]),
        mac="0a:ee:f5:00:00:02",
    ),
    "tx3": Transmitter(
        name="TX3",
        position=np.array([1.8529, 2.2096, 0.5162]),
        mac="0a:ee:f5:00:00:03",
    ),
    "tx4": Transmitter(
        name="TX4",
        position=np.array([-1.0005, 1.0467, 0.5219]),
        mac="0a:ee:f5:00:00:04",
    ),
}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return vector / norm


def array_normal(array: ReceiverArray) -> np.ndarray:
    right = normalize(array.right)
    up = normalize(array.up)

    return normalize(np.cross(right, up))


def antenna_positions(array: ReceiverArray) -> np.ndarray:
    right = normalize(array.right)
    up = normalize(array.up)

    col_offsets = (
        np.arange(array.n_cols) - (array.n_cols - 1) / 2
    ) * array.spacing_horizontal

    row_offsets = (
        np.arange(array.n_rows) - (array.n_rows - 1) / 2
    ) * array.spacing_vertical

    positions = []

    for row_offset in row_offsets:
        for col_offset in col_offsets:
            position = (
                array.center
                + col_offset * right
                + row_offset * up
            )
            positions.append(position)

    return np.asarray(positions)


def local_direction_to_target(
    array: ReceiverArray,
    target_position: np.ndarray,
) -> np.ndarray:
    direction_global = normalize(target_position - array.center)

    right = normalize(array.right)
    up = normalize(array.up)
    normal = array_normal(array)

    return np.array([
        np.dot(direction_global, right),
        np.dot(direction_global, up),
        np.dot(direction_global, normal),
    ])


def true_azimuth_deg(
    array: ReceiverArray,
    target_position: np.ndarray,
) -> float:
    local_direction = local_direction_to_target(array, target_position)

    right_component = local_direction[0]
    normal_component = local_direction[2]

    angle_rad = np.arctan2(right_component, normal_component)

    return float(np.rad2deg(angle_rad))


def true_elevation_deg(
    array: ReceiverArray,
    target_position: np.ndarray,
) -> float:
    local_direction = local_direction_to_target(array, target_position)

    up_component = local_direction[1]
    horizontal_norm = np.linalg.norm([
        local_direction[0],
        local_direction[2],
    ])

    angle_rad = np.arctan2(up_component, horizontal_norm)

    return float(np.rad2deg(angle_rad))