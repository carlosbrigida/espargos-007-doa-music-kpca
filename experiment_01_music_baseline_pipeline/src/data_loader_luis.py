from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


MAC_PREFIX = bytes([0x0A, 0xEE, 0xF5])

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
    pos = tf.ensure_shape(tf.io.parse_tensor(record["pos"], out_type=tf.float64), (3,))
    rssi = tf.ensure_shape(tf.io.parse_tensor(record["rssi"], out_type=tf.float32), (4, 2, 4))
    time = tf.ensure_shape(tf.io.parse_tensor(record["time"], out_type=tf.float64), ())
    mac = tf.ensure_shape(tf.io.parse_tensor(record["mac"], out_type=tf.string), ())

    return csi, pos, time, rssi, mac


def mac_filter(csi, pos, time, rssi, mac):
    return tf.strings.substr(mac, 0, 3) == MAC_PREFIX


def weight_csi_with_rssi(csi, pos, time, rssi, mac):
    csi = tf.cast((10 ** (rssi / 20))[:, :, :, tf.newaxis], tf.complex64) * csi
    return csi, pos, time, rssi, mac


class EspargosLuisTFRecordLoader:
    def __init__(self, tfrecord_files: Iterable[str]):
        self.tfrecord_files = [str(Path(file)) for file in tfrecord_files]

    def load(self) -> dict:
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_files)

        dataset = raw_dataset.map(parse_espargos_record)
        dataset = dataset.filter(mac_filter)
        dataset = dataset.map(weight_csi_with_rssi)

        csi_list = []
        mac_list = []
        pos_list = []
        rssi_list = []
        time_list = []

        for csi, pos, time, rssi, mac in dataset.batch(1000):
            csi_fdomain = np.fft.ifftshift(csi.numpy(), axes=-1)

            csi_list.append(csi_fdomain)
            pos_list.append(pos.numpy())
            rssi_list.append(rssi.numpy())
            time_list.append(time.numpy())
            mac_list.extend(mac.numpy())

        return {
            "csi": np.concatenate(csi_list).astype(np.complex64),
            "mac": np.asarray(mac_list),
            "pos": np.concatenate(pos_list).astype(np.float64),
            "rssi": np.concatenate(rssi_list).astype(np.float32),
            "time": np.concatenate(time_list).astype(np.float64),
        }