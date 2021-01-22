from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_provider
import losses
import models
import os

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim


# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0, 'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97, 'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('batch_size', 5, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'How many preprocess threads to use.')
tf.app.flags.DEFINE_string('train_dir',
'/path/to/save/train/files',
                           '''Directory where to write event logs and checkpoints'''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 1, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('seq_length', 500, 'Number of batches to run.')
tf.app.flags.DEFINE_string('train_device', '/gpu:0', 'Device to train with.')
tf.app.flags.DEFINE_string('model', 'audio',
                           '''Which model is going to be used: audio, video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/path/to/tfrecords',
                           'The tfrecords directory.')

def train(data_folder):
    
    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        audio_frames, ground_truth, _ = data_provider.get_split(data_folder, True,
                                                                'train', FLAGS.batch_size,
                                                                seq_length=FLAGS.seq_length)
        
        # Define model graph.
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=True):
            prediction = models.get_model(audio_frames)

        for i, name in enumerate(['arousal', 'valence']):
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,))

            loss = losses.concordance_cc(pred_single, gt_single)
            tf.summary.scalar('losses/{} loss'.format(name), loss)

            mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            tf.summary.scalar('losses/mse {} loss'.format(name), mse)

            tf.losses.add_loss(loss / 2.)

        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate,momentum = 0.1)

        init_fn = None
        with tf.Session(graph=g) as sess:
            if FLAGS.pretrained_model_checkpoint_path:
                audio_vtr = slim.get_variables()
                audio_saver = tf.train.Saver(audio_vtr)

                def init_fn(sess):
                    audio_saver.restore(sess, FLAGS.both_model)

            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)

            sums = tf.summary.merge_all()
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                FLAGS.train_dir,
                                init_fn=init_fn,
                                save_summaries_secs=60,
                                save_interval_secs=300,
                                summary_op=sums)


if __name__ == '__main__':
    train(FLAGS.dataset_dir)
