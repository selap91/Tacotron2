# code based : https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/tacotron.py

import tensorflow as tf
from utils.symbols import symbols
from utils.infolog import log
from .helpers import TacoTrainingHelper, TacoTestHelper
from .modules import *
from .dynamic_decoder import dynamic_decode
from .custom_decoder import CustomDecoder
from .attention_wrapper import MyAttentionWrapper
from .attentions import LocationBasedAttention, HybridAttention


class Tacotron2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, attention_style=None):
        """
        Initializes the model for inference

        Args:
            - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
              steps in the input time series, and values are character IDs
            - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
            - mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
            - linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
            of steps in the output time series, F is num_freq, and values are entries in the linear
            spectrogram. Only needed for training.
            - attention_style: select the attention mechanism style. if 'None' is to select 'LocationBasedAttention'.
            if not 'None' is to select 'HybridAttention'
        """
        with tf.variable_scope('inference') as scope:
            is_training = mel_targets is not None
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            # Char embeddings
            embedding_table = tf.get_variable(
                'inputs_embedding', [len(symbols), hp.char_embed_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

            ###########
            # Encoder #
            ###########
            # 3 conv layers
            enc_conv_outputs = enc_3convs(embedded_inputs, is_training)
            # Bi-directional LSTM
            encoder_outputs = enc_bi_LSTM(enc_conv_outputs, input_lengths, is_training, 'enc_bi_LSTM')


            ###########
            # Decoder #
            ###########
            # Define the 'style' attention mechanism
            if attention_style is not None:
                attention_mechanism = HybridAttention(hp.attention_size, encoder_outputs)
            else:
                attention_mechanism = LocationBasedAttention(hp.attention_size, encoder_outputs)

            # AttentionWrapper: wrapping (pre-net, 2 uni-directional LSTM, attention, linear-projection)
            # Output: initial predicted mel-frame
            attention_decoder = MyAttentionWrapper(
                dec_uni_LSTM(is_training, layers=2),
                attention_mechanism,
                alignment_history=True,
                output_attention=False)

            decoder_init_state = attention_decoder.zero_state(batch_size=batch_size, dtype=tf.float32)

            # Define helper, predict 'stop-token' in helper.
            if is_training:
                helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
            else:
                helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

            # Decoding
            (decoder_output, _), final_decoder_state, self.stop_error = dynamic_decode(
                CustomDecoder(attention_decoder, helper, decoder_init_state),
                impute_finished=True)

            # 5 convolution layers Post-net : predict residual features. Add to initial mel prediction
            residual_features = dec_postnet(decoder_output, is_training, scope='dec_postnet')
            residual_features = projection(residual_features, hp.num_mels, scope='residual_projection')

            # MEL-SPECTROGRAM
            mel_outputs = decoder_output + residual_features

            # Grab alignments from the final decoder state
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_output = decoder_output
            self.alignments = alignments
            self.mel_outputs = mel_outputs
            self.mel_targets = mel_targets
            log('Initialized Tacotron model. Dimensions: ')
            log('  embedding:               {}'.format(embedded_inputs.shape[-1]))
            log('  enc conv out:            {}'.format(enc_conv_outputs.shape[-1]))
            log('  encoder out:             {}'.format(encoder_outputs.shape[-1]))
            log('  decoder out:             {}'.format(decoder_output.shape[-1]))
            log('  residual out:            {}'.format(residual_features.shape[-1]))
            log('  mel out:                 {}'.format(mel_outputs.shape[-1]))


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams

            # Compute loss of predictions before postnet
            before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
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
            hp = self._hparams
            if hp.decay_learning_rate:
                self.decay_steps = hp.decay_steps
                self.decay_rate = hp.decay_rate
                self.learning_rate = self._learning_rate_decay(hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)

            self.optimize = tf.train.AdamOptimizer(self.learning_rate,
                                               hp.adam_beta1,
                                               hp.adam_beta2,
                                               hp.adam_epsilon).minimize(self.loss,
                                                                         global_step=global_step)

    def _learning_rate_decay(self, init_lr, global_step):
        # Exponential decay starting after 50,000 iterations
        # We won't drop learning rate below 10e-5
        hp = self._hparams
        step = tf.cast(global_step + 1, dtype=tf.float32)
        if tf.greater(step, self.decay_steps) == True:
            lr = tf.train.exponential_decay(init_lr,
                                            global_step - self.decay_steps + 1,
                                            self.decay_steps,
                                            self.decay_rate,
                                            name='exponential_decay')
            return max(hp.final_learning_rate, lr)
        return init_lr