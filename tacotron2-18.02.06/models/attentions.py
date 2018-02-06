from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from hparams import hparams as hp
from .attention_wrapper import _BaseAttentionMechanism


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class HybridAttention(_BaseAttentionMechanism):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 probability_fn=None,
                 score_mask_value=tf.float32.min,
                 name='LocationBasedAttention'):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options include
                @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(HybridAttention, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name='query_layer', use_bias=False),
            memory_layer=layers_core.Dense(
                num_units, name='memory_layer', use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._name = name
        self.v = tf.get_variable("hybrid_attention_v", [num_units], dtype=tf.float32)
        # self.pre_location_filter = tf.get_variable("pre_location_filter",
        #                                           [Config.AttentionConvKernelSize, 1, Config.AttentionConvFilterSize],
        #                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.location_f_weight = tf.get_variable("hybrid_location_f_weight",
                                                 [32, num_units],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                 dtype=tf.float32)

    def __call__(self, query, previous_alignments):
        # processing query(cell output), shape [N, out_dim] -> [N, num_units] -> [N, 1, num_units]
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)

        # alignments shape [N, T_in] -> [N, T_in, 1] -> [N, T_in, 32] -> [N*T_in, 32]
        #                                                               -> [N*T_in, num_units] -> [N, T_in, num_units]
        previous_alignments = tf.expand_dims(previous_alignments, -1)
        location_f = tf.layers.conv1d(previous_alignments, filters=32, kernel_size=31, padding="SAME",
                                      name="hybrid_location_features")
        location_f = tf.reshape(location_f, shape=[-1, 32])
        location_f = tf.matmul(location_f, self.location_f_weight)
        location_f = tf.reshape(location_f, shape=[hp.batch_size, -1, self._num_units])

        # Calculate 'score'
        score = tf.reduce_sum(self.v * tf.tanh(self.keys + processed_query + location_f), [2])
        # Calculate 'alignment' using softmax function.
        alignments = self._probability_fn(score, previous_alignments)
        return alignments



class LocationBasedAttention(_BaseAttentionMechanism):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 probability_fn=None,
                 score_mask_value=tf.float32.min,
                 name='LocationBasedAttention'):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options include
                @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(LocationBasedAttention, self).__init__(
                query_layer=layers_core.Dense(
                        num_units, name='query_layer', use_bias=False),
                memory_layer=layers_core.Dense(
                        num_units, name='memory_layer', use_bias=False),
                memory=memory,
                probability_fn=wrapped_probability_fn,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value,
                name=name)
        self._num_units = num_units
        self._name = name
        self.v = tf.get_variable("location_attention_v", [num_units], dtype=tf.float32)
        # self.pre_location_filter = tf.get_variable("pre_location_filter",
        #                                           [Config.AttentionConvKernelSize, 1, Config.AttentionConvFilterSize],
        #                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.location_f_weight = tf.get_variable("location_location_f_weight",
                                                 [32, num_units],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                 dtype=tf.float32)

    def __call__(self, query, previous_alignments):
        # processing query(cell output), shape [N, out_dim] -> [N, num_units] -> [N, 1, num_units]
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)

        # alignments shape [N, T_in] -> [N, T_in, 1] -> [N, T_in, 32] ->
        #                                        [N*T_in, 32] -> [N*T_in, num_units] -> [N, T_in, num_units]
        previous_alignments = tf.expand_dims(previous_alignments, -1)
        location_f = tf.layers.conv1d(previous_alignments, filters=32, kernel_size=31, padding="SAME",
                                      name="location_location_features")
        location_f = tf.reshape(location_f, shape=[-1, 32])
        location_f = tf.matmul(location_f, self.location_f_weight)
        location_f = tf.reshape(location_f, shape=[hp.batch_size, -1, self._num_units])

        # Calculate 'score'
        score = tf.reduce_sum(self.v * tf.tanh(processed_query + location_f), [2])
        # Calculate 'alignment' using softmax function.
        alignments = self._probability_fn(score, previous_alignments)
        return alignments
