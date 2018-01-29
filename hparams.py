import tensorflow as tf

SCALE_FACTOR = 1

def f(num):
    return num // SCALE_FACTOR

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
  # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
  cleaners = 'english_cleaners',

  # Audio:
  num_mels = 80,
  num_freq = 1025,
  sample_rate = 20000,
  frame_length_ms = 50,
  frame_shift_ms = 12.5,
  preemphasis = 0.97,
  min_level_db = -100,
  ref_level_db = 20,

  # sizes
  char_embed_size = 512,
  attention_size = 128,

  enc_rnn_size = 512,
  enc_3conv_sizes = [f(512), f(512), f(512)],

  dec_rnn_size = 1024,
  dec_prenet_sizes = [f(256), f(256)],
  dec_output_size = 512,

  # Training:
  outputs_per_step = 1,
  batch_size = 8,
  initial_lr = 0.001,
  decay_lr = 0.00001,
  max_iters = 200,
  drop_prob = 0.5,
  zoneout_prob = 0.1,
  use_cmudict=False,
)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)

