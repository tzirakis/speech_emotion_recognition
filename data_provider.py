from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path
from inception_processing import distort_color

slim = tf.contrib.slim

def get_split(dataset_dir, is_training=True, split_name='train', batch_size=32,
              seq_length=100, debugging=False):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.

    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """

    root_path = Path(dataset_dir) / split_name
    paths = [str(x) for x in root_path.glob('*.tfrecords')]

    filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'raw_audio': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'subject_id': tf.FixedLenFeature([], tf.int64),
            'frame': tf.FixedLenFeature([], tf.string),
        }
    )

    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    subject_id = features['subject_id']

    # 640 samples at 16KhZ corresponds to 40ms which is the frequency of
    # annotations.
    raw_audio.set_shape([640])
    label.set_shape([2])

    # Number of threads should always be one, in order to load samples
    # sequentially.
    audio_samples, labels, subject_ids = tf.train.batch(
        [audio_samples, labels, subject_ids], seq_length, num_threads=1, capacity=1000)

    audio_samples = tf.expand_dims(audio_samples, 0)
    labels = tf.expand_dims(labels, 0)

    if is_training:
        audio_samples, labels, subject_ids = tf.train.shuffle_batch(
            [audio_samples, labels, subject_ids], batch_size, 1000, 50, num_threads=1)
    else:
        audio_samples, labels, subject_ids = tf.train.batch(
            [audio_samples, labels, subject_ids], batch_size, num_threads=1, capacity=1000)

    return audio_samples[:, 0, :, :], labels[:, 0, :, :], subject_ids
