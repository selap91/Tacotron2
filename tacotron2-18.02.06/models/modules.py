import tensorflow as tf
from .zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from hparams import hparams as hp


def conv(inputs, channels, kernel_size, is_training, drop_prob, scope, activation=None):
    with tf.variable_scope(scope):
        result = tf.layers.conv1d(
            inputs,
            channels,
            kernel_size,
            padding="same",
            activation=activation)
    result = tf.layers.dropout(result, rate=drop_prob)
    return tf.layers.batch_normalization(result, training=is_training)


def enc_3convs(inputs, is_training, scope=None):
    result = inputs
    drop_rate = hp.drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'enc_3convs'):
        # 3-layers conv
        for i, size in enumerate(hp.enc_3conv_sizes):
            result = conv(result,
                          channels=size,
                          kernel_size=5,
                          activation=tf.nn.relu,
                          is_training=is_training,
                          drop_prob=drop_rate,
                          scope="enc_convs_%d" % (i + 1))
    return result  # [N, T_in, 512]


def enc_bi_LSTM(inputs, input_lengths, is_training, scope=None):
    result = inputs
    with tf.variable_scope(scope or 'enc_bi_LSTM'):
        cell_fw = ZoneoutLSTMCell(hp.enc_rnn_size//2, is_training,
                                  zoneout_factor_cell=hp.zoneout_prob, zoneout_factor_output=hp.zoneout_prob)
        cell_bw = ZoneoutLSTMCell(hp.enc_rnn_size//2, is_training,
                                  zoneout_factor_cell=hp.zoneout_prob, zoneout_factor_output=hp.zoneout_prob)
        outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                           result,
                                                           sequence_length=input_lengths,
                                                           initial_state_fw=None,
                                                           initial_state_bw=None,
                                                           dtype=tf.float32)
    return tf.concat(outputs, axis=2) # Concat forward and backward outputs,     [N, T_in, 256+256]


# In order to introduce output variation, use drop-out for decoder pre-net at inference time too.
def dec_prenet(inputs, layers_size, scope=None):
    result = inputs
    drop_rate = hp.drop_prob
    with tf.variable_scope(scope or 'dec_prenet'):
        for i, size in enumerate(layers_size):
            fc = tf.layers.dense(result, units=size, activation=tf.nn.relu, name='dec_FC_%d' % (i + 1))
            result = tf.layers.dropout(fc, rate=drop_rate, name='dropout_%d' % (i + 1))
    return result


def dec_uni_LSTM(is_training, layers=2):
    rnn_layers = [ZoneoutLSTMCell(hp.dec_rnn_size//2, is_training, zoneout_factor_cell=0.1,
                                  zoneout_factor_output=0.1,
                                  ext_proj=hp.num_mels) for i in range(layers)]
    stacked_LSTM_Cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    return stacked_LSTM_Cell


def dec_postnet(inputs, is_training, scope=None):
    result = inputs
    drop_rate = hp.drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'dec_postnet'):
        activation = tf.nn.tanh
        for i in range(5):
            if i == 4:
                activation = None
            result = conv(result,
                          channels=512,
                          kernel_size=5,
                          activation=activation,
                          is_training=is_training,
                          drop_prob=drop_rate,
                          scope="dec_postnet_%d" % (i + 1))
    return result


def projection(x, shape=512, activation=None, scope=None):
    if scope is None:
        scope = 'linear_projection'

    with tf.variable_scope(scope):
        # if activation==None, this returns a simple linear projection
        # else the projection will be passed through an activation function
        output = tf.contrib.layers.fully_connected(x, shape, activation_fn=activation,
                                                   biases_initializer=tf.zeros_initializer(),
                                                   scope=scope)
        return output

