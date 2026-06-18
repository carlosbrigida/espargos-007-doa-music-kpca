from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


FEATURE_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "mac": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "pos": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "rssi": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "time": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def parse_espargos_record(proto):
    record = tf.io.parse_single_example(proto, FEATURE_DESCRIPTION)

    csi = tf.ensure_shape(
        tf.io.parse_tensor(record["csi"], out_type=tf.complex64),
        (4, 2, 4, 53),
    )

    pos = tf.ensure_shape(
        tf.io.parse_tensor(record["pos"], out_type=tf.float64),
        (3,),
    )

    rssi = tf.ensure_shape(
        tf.io.parse_tensor(record["rssi"], out_type=tf.float32),
        (4, 2, 4),
    )

    time = tf.ensure_shape(
        tf.io.parse_tensor(record["time"], out_type=tf.float64),
        (),
    )

    mac = record["mac"]

    return csi, mac, pos, rssi, time


class EspargosTFRecordLoader:
    """
    Loader for ESPARGOS 007 TFRecord files.
    """

    def __init__(self, tfrecord_files: Iterable[str]):
        self.tfrecord_files = [str(Path(file)) for file in tfrecord_files]

    def load(self) -> dict:
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_files)

        dataset = raw_dataset.map(
            parse_espargos_record,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        csi_list = []
        mac_list = []
        pos_list = []
        rssi_list = []
        time_list = []

        for csi, mac, pos, rssi, time in dataset:
            csi_list.append(csi.numpy())
            mac_list.append(mac.numpy())
            pos_list.append(pos.numpy())
            rssi_list.append(rssi.numpy())
            time_list.append(time.numpy())

        return {
            "csi": np.asarray(csi_list, dtype=np.complex64),
            "mac": np.asarray(mac_list),
            "pos": np.asarray(pos_list, dtype=np.float64),
            "rssi": np.asarray(rssi_list, dtype=np.float32),
            "time": np.asarray(time_list, dtype=np.float64),
        }