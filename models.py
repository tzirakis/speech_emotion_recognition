from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim

def recurrent_model(net, hidden_units=256, number_of_outputs=2):

    with tf.variable_scope("recurrent_model"):
      batch_size, seq_length, num_features = net.get_shape().as_list()

      lstm1 = tf.contrib.rnn.LSTMCell(256,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      lstm2 = tf.contrib.rnn.LSTMCell(256,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

      # We have to specify the dimensionality of the Tensor so we can allocate
      # weights for the fully connected layers.
      outputs, states = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

      net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

      prediction = slim.layers.linear(net, 2)

      return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))


def audio_model(audio_frames):

    with tf.variable_scope("audio_model"):
      batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
      audio_input = tf.reshape(audio_frames, [batch_size,  num_features * seq_length, 1])
    
      net = tf.layers.conv1d(audio_input,64,8,padding = 'same', activation = tf.nn.relu)
      net = tf.layers.max_pooling1d(net,10,10)
      net = slim.dropout(net,0.5)

      net = tf.layers.conv1d(net,128,6,padding = 'same', activation = tf.nn.relu)
      net = tf.layers.max_pooling1d(net,8,8)
      net = slim.dropout(net,0.5)

      net = tf.layers.conv1d(net,256,6,padding = 'same', activation = tf.nn.relu)
      net = tf.layers.max_pooling1d(net,8,8)
      net = slim.dropout(net,0.5)

      net = tf.reshape(net,[batch_size,seq_length,-1]) #256])
    return net

def get_model(audio_frames):
    return recurrent_model(audio_model(audio_frames))
