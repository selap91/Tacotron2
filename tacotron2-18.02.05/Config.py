

# Audio
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20

# Encoder
char_embed_size = 512
enc_3conv_sizes = (512, 512, 512)
enc_rnn_size = 512

# Attention
attention_size = 128

# Decoder
dec_prenet_size = (256, 256)
dec_rnn_size = 1024
dec_out_size = 512

# Training
batch_size = 8
drop_prob = 0.5
zoneout_prob = 0.1
output_per_step = 1
use_decay = True
initial_lr = 10e-3
final_lr = 10e-5
decay_start = 50000
decay_rate = 0.96
reg_weight = 10e-6
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 10e-6
use_cmudict = False