# code based on https://github.com/keithito/tacotron/

import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.rnn import MultiRNNCell
from text.symbols import symbols

from util.infolog import log
from .helpers import *
from .wrappers import *
from .attention import *
from .modules import *
from .outputWrapper import OutputProjectionWrapper


class Tacotron2():
    def __init__(self, hyperpara):
        self._hyperpara = hyperpara


    def initialize(
            self, inputs, input_lengths, mel_targets=None, linear_targets=None):

        # At inference
        with tf.variable_scope('inference'):
            is_training = mel_targets is not None
            hp = self._hyperpara
            batch_size = tf.shape(inputs[0])

            # Char embeddings
            char_embed_table = tf.get_variable(
                'char_embedding', [len(symbols), hp.char_embed_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))

            char_embedded_inputs = tf.nn.embedding_lookup(char_embed_table, inputs) # [N, T_in, 512]

            # Encoder #
            encoder_outputs = encoder(char_embedded_inputs, input_lengths, is_training, scope='encoder') # [N, T_in, 512]

            # Decoder #
            # Pre-net + double-LSTM
            #dec_LSTM = ZoneoutWrapper(LSTMCell(hp.dec_rnn_size), hp.zoneout_prob, is_training)
            dec_LSTM = LSTMCell(hp.dec_rnn_size)
            dec_double_LSTM = MultiRNNCell([dec_LSTM]*2, state_is_tuple=True)
            attention_cell = AttentionWrapper(DecoderPrenetWrapper(dec_double_LSTM, is_training),
                                              BahdanauAttention(hp.attention_size, encoder_outputs),
                                              alignment_history=True,
                                              output_attention=False)
                                # if 'output_attention' is True, the output is attention output.

            #concat_cell = ConcatOutputAndAttentionWrapper(attention_cell) # concat attention context vector with LSTM outputs

            # linear-projection : predict frame outputs
            frame_out_cell = OutputProjectionWrapper(attention_cell, hp.num_mels) # decoder-linear-projection, [N, T_out, 80]
            decoder_init_state = frame_out_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # helpers
            if is_training:
                helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, 1)
            else:
                # predict 'stop token'
                stop_token = tf.reshape(frame_out_cell._concats, shape=[batch_size, -1])
                stop_token = tf.layers.dense(stop_token, 1, activation=tf.nn.sigmoid) # [N, 1]
                stop_token = tf.cast(stop_token > 0.5, dtype=tf.bool)

                helper = TacoTestHelper(batch_size, hp.num_mels, stop_token, 1)

            (frame_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(frame_out_cell, helper, decoder_init_state),
                maximum_iterations=hp.max_iters) # [N, T_out, 80]

            # five convolution layers Post-net : residual features add to initial prediction
            postnet_outputs = dec_postnet(frame_outputs, is_training, hp.drop_prob, scope='dec_postnet')

            mel_outputs = frame_outputs + postnet_outputs
            #mel_outputs = tf.reshape(mel_outputs, [batch_size, hp.num_mels])

            # Grab alignments from the final decoder state:
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.mel_initial_outputs = frame_outputs
            self.mel_outputs = mel_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            log('Initialized Tacotron2 model. Dimensions: ')
            log('  embedding:               %d' % char_embedded_inputs.shape[-1])
            log('  encoder out:             %d' % encoder_outputs.shape[-1])
            log('  attention out:           %d' % attention_cell.output_size)
            log('  frame_out cell out:        %d' % frame_out_cell.output_size)
            log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            self.before_post_loss = tf.reduce_mean(tf.square(self.mel_targets - self.mel_initial_outputs))
            self.after_post_loss = tf.reduce_mean(tf.square(self.mel_targets - self.mel_outputs))
            self.mel_loss = self.before_post_loss + self.after_post_loss

    def set_lr(self, is_decay):
        if is_decay:
            return hp.decay_lr
        else:
            return hp.initial_lr

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hyperpara
            self.learning_rate = tf.cond(global_step < 50000, lambda: self.set_lr(False), lambda: self.set_lr(True))

            optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.9, 0.999, 1e-6)
            gradients, variables = zip(*optimizer.compute_gradients(self.mel_loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)
