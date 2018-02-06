# Tacotron2

An implementation of Tacotron2 (excluding WaveNet-vocoder) in TensorFlow.

2018-01-30 incompleted,

2018-02-04 incompleted,

2018-02-05 incompleted,

2018-02-06 complete !

# Attentions:
> attentions.py

There are two styles of attention-mechanism. (1.LocationBasedAttention, 2.HybridAttention)

1.LocationBasedAttention

Calculate the score using (query, alignments).

    score = tf.reduce_sum(self.v * tf.tanh(processed_query + location_f), [2])

2.HybridAttention

Calculate the score using (processed-memory, query, alignments)

    score = tf.reduce_sum(self.v * tf.tanh(self.keys + processed_query + location_f), [2])
		
# MyAttentionWrapper:

> attention_wrapper.py

When create 'MyAttentionWrapper', using just RNNcells not some 'pre-net wrapper'.

So there are 6 steps
- Step 1: Passed through a 'Pre-net'.

		"The prediction from the previous time step is first passed through a small "pre-net" contatining 2 fully connected layers of 256 hidden ReLU units."
- Step 2: Mix the `inputs` and previous step's `attention` output via `cell_input_fn`.

 		"The pre-net output and attention context vector are concatenated and ..."
- Step 3: Call the wrapped `cell` with this input and its previous state.

 		"...and passed through a stack of 2 uni-directional LSTM layers with 1024 units."
- Step 4: Concat 'LSTM_output' with 'context_vector(attention)'

 		"The concatenation of the LSTM output and the attention context vector is then..."
- Step 5: Concatted output projected through linear-transform.

 		"...then projected through a linear transform to produce a prediction of the target spectrogram frame."
- Step 6: Calculate the score, alignments and attention.

# 
So many references from https://github.com/Rayhane-mamah/Tacotron-2
