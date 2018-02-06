import tensorflow as tf 


# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
	cleaners='english_cleaners',

	#Audio
	num_mels=80,
	num_freq=1025,
	sample_rate=24000,
	frame_length_ms=50,
	frame_shift_ms=12.5,
	preemphasis=0.97,
	min_level_db=-100,
	ref_level_db=20,
	cmu_dict=False,

	#Model
	outputs_per_step = 1,
	parameter_init = 0.5,
	sharpening_factor = 1.0,
	max_decode_length = None,
	num_classes = None,
	time_major = False,
	hidden_dim = 128,

	# Encoder
	char_embed_size=512,
	enc_3conv_sizes=[512, 512, 512],
	enc_rnn_size=512,

	# Attention
	attention_size=128,

	# Decoder
	dec_prenet_sizes=[256, 256],
	dec_rnn_size=1024,

	#Training
	batch_size=64,
	drop_prob = 0.5,
	zoneout_prob = 0.1,
	reg_weight = 10e-6,
	decay_learning_rate = True,
	decay_steps = 50000,
	decay_rate = 0.96,
	initial_learning_rate = 10e-3,
	final_learning_rate = 10e-5,
	adam_beta1 = 0.9,
	adam_beta2 = 0.999,
	adam_epsilon = 10e-6,
	)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
