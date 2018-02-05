import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, \
    AttentionWrapperState
import Config as cf

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

class LocationBasedAttention(_BaseAttentionMechanism):

    def __init__(self, num_units, memory, memory_sequence_length=None, scope="LocationBasedAttention"):
        self._name = scope
        self._num_units = num_units

        with tf.variable_scope(scope):
            query_layer = tf.layers.Dense(num_units, name="location_query_layer", use_bias=False)
            memory_layer = tf.layers.Dense(num_units, name="location_memory_layer", use_bias=False)
            self.v = tf.get_variable("location_attention_v", [num_units], dtype=tf.float32)
            #self.pre_location_filter = tf.get_variable("pre_location_filter",
            #                                           [Config.AttentionConvKernelSize, 1, Config.AttentionConvFilterSize],
            #                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.location_f_weight = tf.get_variable("location_location_f_weight",
                                                       [32, num_units],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                       dtype=tf.float32)
        wrapped_probability_fn = lambda score, _: tf.nn.softmax(score)

        super(LocationBasedAttention, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=float("-inf"),
            name=self._name)

    def __call__(self, query, state):
        # processing query(cell output), shape [N, out_dim] -> [N, num_units] -> [N, 1, num_units]
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)

        # Alignments shape [N, T_in] -> [N, T_in, 1] -> [N, T_in, 32] -> [N*T_in, 32]
        #                                                               -> [N*T_in, num_units] -> [N, T_in, num_units]
        previous_alignments = tf.expand_dims(state, -1)
        location_f = tf.layers.conv1d(previous_alignments, filters=32, kernel_size=31, padding="SAME", name="location_features")
        location_f = tf.reshape(location_f, shape=[-1, 32])  # [N*T_in, 32] -> [N*T_in, 128]
        location_f = tf.matmul(location_f, self.location_f_weight)
        location_f = tf.reshape(location_f, shape=[cf.batch_size, -1, 128])

        # Calculate 'score'
        score = tf.reduce_sum(self.v * tf.tanh(processed_query + location_f), [2])
        # Calculate 'alignment' using softmax function.
        alignments = self._probability_fn(score, previous_alignments)
        return alignments


class HybridAttention(_BaseAttentionMechanism):

    def __init__(self, num_units, memory, memory_sequence_length=None, scope="HybridAttention"):
        self._name = scope
        self._num_units = num_units

        with tf.variable_scope(scope):
            query_layer = tf.layers.Dense(num_units, name="hybrid_query_layer", use_bias=False)
            memory_layer = tf.layers.Dense(num_units, name="hybrid_memory_layer", use_bias=False)
            self.v = tf.get_variable("hybrid_attention_v", [num_units], dtype=tf.float32)
            # self.pre_location_filter = tf.get_variable("pre_location_filter",
            #                                           [Config.AttentionConvKernelSize, 1, Config.AttentionConvFilterSize],
            #                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.location_f_weight = tf.get_variable("hybrid_location_f_weight",
                                                       [32, num_units],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                       dtype=tf.float32)
        wrapped_probability_fn = lambda score, _: tf.nn.softmax(score)

        super(HybridAttention, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=float("-inf"),
            name=self._name)

    def __call__(self, query, state):
        # processing query(cell output), shape [N, out_dim] -> [N, num_units] -> [N, 1, num_units]
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)

        # alignments shape [N, T_in] -> [N, T_in, 1] -> [N, T_in, 32] -> [N*T_in, 32]
        #                                                               -> [N*T_in, num_units] -> [N, T_in, num_units]
        previous_alignments = tf.expand_dims(state, -1)
        location_f = tf.layers.conv1d(previous_alignments, filters=32, kernel_size=31, padding="SAME",
                                      name="location_features")
        location_f = tf.reshape(location_f, shape=[-1, 32])  # [N*T_in, 32] -> [N*T_in, 128]
        location_f = tf.matmul(location_f, self.location_f_weight)
        location_f = tf.reshape(location_f, shape=[cf.batch_size, -1, 128])

        # Calculate 'score'
        score = tf.reduce_sum(self.v * tf.tanh(self.keys + processed_query + location_f), [2])
        # Calculate 'alignment' using softmax function.
        alignments = self._probability_fn(score, previous_alignments)
        return alignments, alignments


class MyAttentionWrapperState(
    namedtuple("MyAttentionWrapperState",
                           ("cell_state", "attention", "attention_history", "time", "alignments",
                            "alignment_history"))):

  def clone(self, **kwargs):
    return super(MyAttentionWrapperState, self)._replace(**kwargs)


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = tf.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = tf.matmul(expanded_alignments, attention_mechanism.values)
  context = tf.squeeze(context, [1])

  #if attention_layer is not None:
  #  attention = attention_layer(tf.concat([cell_output, context], 1))
  #else:
  attention = context

  return attention, alignments



class MyAttentionWrapper(RNNCell):
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=True,
                 scope="MyAttentionWrapper"):

        super(MyAttentionWrapper, self).__init__(name=scope+"_AttentionWrapper")
        with tf.variable_scope(scope):
            self.attention_layer = tf.layers.Dense(attention_layer_size, name="attention_layer", use_bias=False)

        self.attention_layer_size = attention_layer_size
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.cell_input_fn = (lambda inputs, attention: tf.concat([inputs, attention], -1))
        self.use_alignment_history = alignment_history

    @property
    def output_size(self):
        return self.cell.output_size

    @property
    def state_size(self):
        return MyAttentionWrapperState(
            cell_state=self.cell.state_size,
            time=tf.TensorShape([]),
            attention=self.attention_layer_size,
            attention_history=(),
            alignments=self.attention_mechanism.alignments_size, # max_time
            alignment_history=())  # sometimes a TensorArray


    def zero_state(self, batch_size, dtype):
        cell_state = self.cell.zero_state(batch_size, dtype)
        return self.init_state(batch_size, cell_state)

    def init_state(self, batch_size, init_cell_state):
        cell_state = tf.contrib.framework.nest.map_structure(lambda s: tf.identity(s, name="checked_cell_state"), init_cell_state)

        return MyAttentionWrapperState(
            cell_state=cell_state,
            time=tf.zeros([], dtype=tf.int32),
            #attention=tf.zeros(shape=[batch_size, self.attention_layer_size], dtype=tf.float32),
            attention=_zero_state_tensors(self.attention_layer_size, batch_size, tf.float32),
            attention_history=tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
            alignments=self.attention_mechanism.initial_alignments(batch_size, tf.float32),
            alignment_history=tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))

    def call(self, inputs, state):
        cell_inputs = self.cell_input_fn(inputs, state.attention)
        #cell_state = state.cell_state
        (cell_output, LSTM_output), next_cell_state = self.cell(cell_inputs, state)
        #cell_output, LSTM_output, next_cell_state = self.cell(inputs, state)
        cell_output = tf.identity(cell_output, name="checked_cell_output")

        previous_alignment = state.alignments
        previous_alignment_history = state.alignment_history
        previous_attention_history = state.attention_history

        attention_mechanism = self.attention_mechanism
        attention, alignments = _compute_attention(attention_mechanism, cell_output, previous_alignment, self.attention_layer)
        alignment_history = previous_alignment_history.write(state.time, alignments) if self.use_alignment_history else ()
        attention_history = previous_attention_history.write(state.time, attention)

        next_state = MyAttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_history=attention_history,
            alignments=alignments,
            alignment_history=alignment_history)

        return cell_output, next_state, LSTM_output