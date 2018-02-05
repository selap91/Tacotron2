from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import audio
import os
import numpy as np 


def build_from_path(input_dir, out_dir, n_jobs=4, tqdm=lambda x: x):
	"""
	Preprocesses the Lj speech dataset from a gven input path to a given output directory

	Args:
		- in_dir: input directory that contains the files to prerocess
		- out_dir: output drectory of the preprocessed Lj dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for 
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(parts[0]))
			text = parts[2]
			futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
			index += 1
	return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- out-dir: the directory to write the spectograms into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file

	Returns:
		- A tuple: (mel_filename, n_frames, text)
	"""

	# Load the audio as numpy array
	wav = audio.load_wav(wav_path)

	# Compute the linear-scale spectrogram from the wav to calculate n_frames
	spectrogram = audio.spectrogram(wav).astype(np.float32)
	n_frames = spectrogram.shape[1]

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

	# Write the spectrogram to disk
	mel_filename = 'ljspeech-mel-{:05d}.npy'.format(index)
	np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (mel_filename, n_frames, text)