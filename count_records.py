import tensorflow as tf

dataset = tf.data.TFRecordDataset(
    "espargos-0007-human-helmet-standing-center-1.tfrecords"
)

count = 0

for _ in dataset:
    count += 1

print("Total:", count)