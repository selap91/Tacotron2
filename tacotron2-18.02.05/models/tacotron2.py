
import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.rnn import MultiRNNCell
from utils.symbols import symbols
from utils.infolog import log
from .helpers import TacoTrainingHelper, TacoTestHelper
from .modules import *
from .rnn_wrappers import TacotronDecoderWrapper
from .dynamic_decoder import dynamic_decode
from .custom_decoder import CustomDecoder
from .LocationSensitiveAttention import MyAttentionWrapper, LocationBasedAttention, HybridAttention


class Tacotron2():
    def __init__(self, hyperpara):
        self._hyperpara = hyperpara


    def initialize(
            self, inputs, input_lengths, mel_targets=None, linear_targets=None, style=True):
        '''
        Args:
            - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
              steps in the input time series, and values are character IDs.
              It means the input text.
            - input_lenghts: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
            It means the length of each data in the batch.
            - mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
            It means MEL-SPECTROGRAM representation of input audio.
            - linear_targets: For tacotron1, just ignore.
        '''

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

            ##########
            # Encoder
            ##########
            encoder_outputs = encoder(char_embedded_inputs, input_lengths, is_training, scope='encoder') # [N, T_in, 512]

            ##########
            # Decoder
            ##########
            # Wrapping decoder_prenet with 2layer uni-directional LSTM
            prenet_wrapped_cell = TacotronDecoderWrapper(dec_uni_LSTM(is_training, layers=2), is_training)

            # Create the 'style' attention mechanism
            if style:
                attention_mechanism = HybridAttention(cf.attention_size, encoder_outputs)
            else:
                attention_mechanism = LocationBasedAttention(cf.attention_size, encoder_outputs)

            # Wrapping wrapped RNN with Attention mechanism
            attention_wrapped_cell = MyAttentionWrapper(prenet_wrapped_cell, attention_mechanism, attention_layer_size=cf.dec_out_size)
            decoder_init_state = attention_wrapped_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # Define helper, predict 'stop-token' in helper.
            if is_training:
                helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, cf.output_per_step)
            else:
                helper = TacoTestHelper(batch_size, hp.num_mels, cf.output_per_step)

            # Decode
            (frame_outputs, _), final_decoder_state, self.stop_error = dynamic_decode(
                CustomDecoder(attention_wrapped_cell, helper, decoder_init_state),
                impute_finished=True)

            # 5 convolution layers Post-net : predict residual features, add to initial prediction
            residual_features = dec_postnet(frame_outputs, is_training, scope='dec_postnet')
            residual_features = projection(residual_features, hp.num_mels, scope='residual_projection')

            # MEL-SPECTROGRAM
            mel_outputs = frame_outputs + residual_features

            # Grab alignments from the final decoder state [N, T_out, ?] or [N, ?, T_out]
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.mel_initial = frame_outputs # [N, T_out, 80]
            self.mel_outputs = mel_outputs # [N, T_out, 80]
            self.mel_targets = mel_targets # [N, T_out, 80]
            self.alignments = alignments
            log('Initialized Tacotron2 model. Dimensions: ')
            log('  embedding:                  %d' % char_embedded_inputs.shape[-1])
            log('  encoder out:                %d' % encoder_outputs.shape[-1])
            log('  pre-net wrapped cell out:   %d' % prenet_wrapped_cell.output_size)
            log('  attention wrapped cell out: %d' % attention_wrapped_cell.output_size)
            log('  decoder out (1 frame):      %d' % mel_outputs.shape[-1])


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hyperpara

            # Compute loss of predictions before postnet
            before = tf.losses.mean_squared_error(self.mel_targets, self.mel_initial)
            # Compute loss after postnet
            after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)

            # Get all trainable variables
            all_vars = tf.trainable_variables()
            # Compute the regularization term
            regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * hp.reg_weight

            # Compute final loss term
            self.before_loss = before
            self.after_loss = after
            self.regularization_loss = regularization

            self.loss = self.before_loss + self.after_loss + self.regularization_loss + self.stop_error


    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hyperpara
            if hp.use_decay:
                self.decay_steps = hp.decay_start
                self.decay_rate = hp.decay_rate
                self.learning_rate = self._learning_rate_decay(hp.initial_lr, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_lr)

            self.optimize = tf.train.AdamOptimizer(self.learning_rate,
                                                   hp.adam_beta1,
                                                   hp.adam_beta2,
                                                   hp.adam_epsilon).minimize(self.loss,
                                                                             global_step=global_step)


    def _learning_rate_decay(self, init_lr, global_step):
        # Exponential decay starting after 50,000 iterations
        # We won't drop learning rate below 10e-5
        hp = self._hyperpara
        step = tf.cast(global_step + 1, dtype=tf.float32)
        if tf.greater(step, self.decay_steps) == True:
            lr = tf.train.exponential_decay(init_lr,
                                            global_step - self.decay_steps + 1, #
                                            self.decay_steps,
                                            self.decay_rate,
                                            name='exponential_decay')
            return max(hp.final_learning_rate, lr)
        return init_lr


