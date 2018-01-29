# code based on https://github.com/keithito/tacotron/

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, RNNCell
from hparams import hparams as hp


class ZoneoutWrapper(RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
    super(ZoneoutWrapper, self).__init__()
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(zoneout_prob, float) and
          not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
    self._cell = cell
    self._zoneout_prob = zoneout_prob
    self._seed = seed
    self.is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    output, new_state = self._cell(inputs, state, scope)

    if self.is_training:
      new_state = (1 - self._zoneout_prob) * new_state + (self._zoneout_prob * state)

    return output, new_state


def enc_3convs(inputs, is_training, drop_prob, scope=None):
    result = inputs # [N, T_in, 512]
    drop_rate = drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'enc_3convs'):
        # 3-layers conv
        for i, size in enumerate(hp.enc_3conv_sizes):
            result = conv(1,
                          result,
                          channels=size,
                          kernel_size=5,
                          activation=tf.nn.relu,
                          is_training=is_training,
                          drop_prob=drop_rate,
                          scope="enc_convs_%d" % (i + 1))
    return result         # [N, T_in, 512]


def enc_bi_LSTM(inputs, input_lengths, num_units, is_training, zoneout_prob, scope=None):
    result = inputs
    with tf.variable_scope(scope or 'enc_bi_LSTM'):
        # single bi-directional LSTM 512
        cell_fw, cell_bw = LSTMCell(num_units), LSTMCell(num_units)  # create each direction 256 cell
        #cell_fw = ZoneoutWrapper(LSTMCell(num_units), zoneout_prob, is_training)
        #cell_bw = ZoneoutWrapper(LSTMCell(num_units), zoneout_prob, is_training)
        outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                           result,
                                                           sequence_length=input_lengths,
                                                           initial_state_fw=None,
                                                           initial_state_bw=None,
                                                           dtype=tf.float32)
    return tf.concat(outputs, axis=2) # Concat forward and backward outputs,     [N, T_in, 256+256]


def encoder(inputs, input_lengths, is_training=True, scope="encoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # encoder 3 convolution layers
        enc_convs_output = enc_3convs(inputs,
                                  is_training=is_training,
                                  drop_prob=hp.drop_prob,
                                  scope="enc_3convs")

        # encoder bi-directional LSTM
        enc_LSTM_outputs = enc_bi_LSTM(enc_convs_output,
                                       input_lengths=input_lengths,
                                       num_units=hp.enc_rnn_size//2,
                                       is_training=is_training,
                                       zoneout_prob=hp.zoneout_prob,
                                       scope="enc_bi_LSTM")

    return enc_LSTM_outputs             # [N, T_in, 512]


def dec_prenet(inputs, layers_size, drop_prob, scope=None):
    result = inputs
    drop_rate = drop_prob
    with tf.variable_scope(scope or 'dec_prenet'):
        for i, size in enumerate(layers_size):
            fc = tf.layers.dense(result, units=size, activation=tf.nn.relu, name='dec_FC_%d' % (i + 1))
            result = tf.layers.dropout(fc, rate=drop_rate, name='dropout_%d' % (i + 1))
    return result              # [N, ?, 256]



def dec_postnet(inputs, is_training, drop_prob, scope=None):
    result = inputs
    drop_rate = drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'dec_postnet'):
        for i in range(5):
            if i != 4:
                activation = tf.nn.tanh
                result = conv(1,
                              result,
                              channels=512,
                              kernel_size=5,
                              activation=activation,
                              is_training=is_training,
                              drop_prob=drop_rate,
                              scope="dec_postnet_%d" % (i + 1))
            else:
                activation = None
                result = conv(1,
                              result,
                              channels=512,
                              kernel_size=5,
                              activation=activation,
                              is_training=is_training,
                              drop_prob=drop_rate,
                              scope="dec_postnet_%d" % (i + 1))
        result = tf.layers.dense(result, 80)

    return result



def conv(dim, inputs, channels, kernel_size, is_training, drop_prob, scope, activation=None):
    with tf.variable_scope(scope):
        if dim == 1:
            result = tf.layers.conv1d(
                inputs,
                channels,
                kernel_size,
                padding="same",
                activation=activation)
        else:
            result = tf.layers.conv2d(
                inputs,
                channels,
                kernel_size,
                padding="same",
                activation=activation)
        result = tf.layers.dropout(result, rate=drop_prob)
        return tf.layers.batch_normalization(result, training=is_training)

