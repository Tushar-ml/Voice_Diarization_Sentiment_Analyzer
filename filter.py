import wave
import deepspeech
import collections
import contextlib
import matplotlib.pyplot as plt 
import sys
import wave
import numpy as np
import webrtcvad
import time 
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emotion_recognition import EmotionRecognizer
from utils import get_best_estimators
import librosa
from pydub import AudioSegment
import pandas as pd
import random,pickle
model = deepspeech.Model('deepspeech-model.pbmm')

stream_context = model.createStream()
#####################################################################################################
# 										Emotion Analyzer 											#										
#####################################################################################################
def get_estimators_name(estimators):
	result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
	return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}
estimators = get_best_estimators(True)
estimators_str, estimator_dict = get_estimators_name(estimators)
features = ["mfcc", "chroma", "mel"]
detector = EmotionRecognizer(estimator_dict['BaggingClassifier'], emotions='angry,happy,neutral'.split(","), features=features, verbose=0)
modelObj = open('model.pkl','rb')
model_emotion = pickle.load(modelObj)
#####################################################################################################

def sentiment_scores(sentence): 
  
	# Create a SentimentIntensityAnalyzer object. 
	sid_obj = SentimentIntensityAnalyzer() 
  
	# polarity_scores method of SentimentIntensityAnalyzer 
	# oject gives a sentiment dictionary. 
	# which contains pos, neg, neu, and compound scores. 
	sentiment_dict = sid_obj.polarity_scores(sentence) 
  
	# decide sentiment as positive, negative and neutral 
	if sentiment_dict['compound'] >= 0.05 : 
		return 'POSITIVE'
  
	elif sentiment_dict['compound'] <= - 0.05 : 
		return 'NEGATIVE' 
  
	else : 
		return 'NEUTRAL'

def read_wave(path):
	"""Reads a .wav file.

	Takes the path, and returns (PCM audio data, sample rate).
	"""
	with contextlib.closing(wave.open(path, 'rb')) as wf:
		num_channels = wf.getnchannels()
		assert num_channels == 1
		sample_width = wf.getsampwidth()
		assert sample_width == 2
		sample_rate = wf.getframerate()
		assert sample_rate in (8000, 16000, 32000, 48000)
		pcm_data = wf.readframes(wf.getnframes())
		return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
	"""Writes a .wav file.

	Takes path, PCM audio data, and sample rate.
	"""
	with contextlib.closing(wave.open(path, 'wb')) as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		wf.writeframes(audio)


class Frame(object):
	"""Represents a "frame" of audio data."""
	def __init__(self, bytes, timestamp, duration):
		self.bytes = bytes
		self.timestamp = timestamp
		self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
	"""Generates audio frames from PCM audio data.

	Takes the desired frame duration in milliseconds, the PCM data, and
	the sample rate.

	Yields Frames of the requested duration.
	"""
	n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
	offset = 0
	timestamp = 0.0
	duration = (float(n) / sample_rate) / 2.0
	while offset + n < len(audio):
		yield Frame(audio[offset:offset + n], timestamp, duration)
		timestamp += duration
		offset += n


def vad_collector(sample_rate, frame_duration_ms,
				  padding_duration_ms, vad, frames):
	"""Filters out non-voiced audio frames.

	Given a webrtcvad.Vad and a source of audio frames, yields only
	the voiced audio.

	Uses a padded, sliding window algorithm over the audio frames.
	When more than 90% of the frames in the window are voiced (as
	reported by the VAD), the collector triggers and begins yielding
	audio frames. Then the collector waits until 90% of the frames in
	the window are unvoiced to detrigger.

	The window is padded at the front and back to provide a small
	amount of silence or the beginnings/endings of speech around the
	voiced frames.

	Arguments:

	sample_rate - The audio sample rate, in Hz.
	frame_duration_ms - The frame duration in milliseconds.
	padding_duration_ms - The amount to pad the window, in milliseconds.
	vad - An instance of webrtcvad.Vad.
	frames - a source of audio frames (sequence or generator).

	Returns: A generator that yields PCM audio data.
	"""
	num_padding_frames = int(padding_duration_ms / frame_duration_ms)
	# We use a deque for our sliding window/ring buffer.
	ring_buffer = collections.deque(maxlen=num_padding_frames)
	# We have two states: TRIGGERED and NOTTRIGGERED. We start in the
	# NOTTRIGGERED state.
	triggered = False

	voiced_frames = []
	for frame in frames:
		is_speech = vad.is_speech(frame.bytes, sample_rate)


		if not triggered:
			ring_buffer.append((frame, is_speech))
			num_voiced = len([f for f, speech in ring_buffer if speech])
			# If we're NOTTRIGGERED and more than 90% of the frames in
			# the ring buffer are voiced frames, then enter the
			# TRIGGERED state.
			if num_voiced > 0.75 * ring_buffer.maxlen:
				triggered = True

				# We want to yield all the audio we see from now until
				# we are NOTTRIGGERED, but we have to start with the
				# audio that's already in the ring buffer.
				for f, s in ring_buffer:
					voiced_frames.append(f)
				ring_buffer.clear()
		else:
			# We're in the TRIGGERED state, so collect the audio data
			# and add it to the ring buffer.
			voiced_frames.append(frame)
			ring_buffer.append((frame, is_speech))
			num_unvoiced = len([f for f, speech in ring_buffer if not speech])
			# If more than 90% of the frames in the ring buffer are
			# unvoiced, then enter NOTTRIGGERED and yield whatever
			# audio we've collected.
			if num_unvoiced > 0.75 * ring_buffer.maxlen:

				triggered = False
				yield b''.join([f.bytes for f in voiced_frames])
				ring_buffer.clear()
				voiced_frames = []
	if triggered:
		sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
	sys.stdout.write('\n')
	# If we have any leftover voiced audio when we run out of input,
	# yield it.
	if voiced_frames:
		yield b''.join([f.bytes for f in voiced_frames])

tracker = 0
timestamp = 0
def transcription_filter(text):
	pass

	''' This Function will be used to write up the results in filename'''
	
def result_writer(audio_file,emotion,result,filename,file=False):
		
		global timestamp
		if file:
			timestamp = 0
		file = open(filename,'a')
		n = 'Neutral'
		if emotion ==  'neutral':
			if result == 'POSITIVE':
				n = 'Positive'
				
			if result == 'NEGATIVE':
				n = 'Negative'
				
			else:
				n = 'Neutral'
					
			
		if emotion == 'happy':
			if result == 'POSITIVE':
				n = 'Positive'
				
			if result == 'NEGATIVE':
				n = 'Positive'
			   
			else:
				n = 'Positive'
				

		if emotion ==  'angry':
			if result == 'POSITIVE':
				n = 'Negative'
			

			if result == 'NEGATIVE':
				n = 'Negative'
				

			else:
				n = 'Negative'
		audio,sr = librosa.load(audio_file,sr=16000)
		length = librosa.get_duration(filename=audio_file)
		audio = audio[200:-200]
		
		new_timestamp = length + timestamp
		a = np.linspace(timestamp,new_timestamp,len(audio))
		print('Writing timestamp files....')
		for i in range(len(audio)):
			file.write(f'{a[i]},{audio[i]},{n}\n')
		timestamp = new_timestamp


previous_segment_length = 0
def main(audio=None,file=False,folder=None,sample_rate=16000,labels=[0,1]):
	global previous_segment_length
	global timestamp
	global tracker
	vad = webrtcvad.Vad()
	if file:
		print(file)
		audio,sr = read_wave(file)
	frames = frame_generator(20, audio, sample_rate)
	frames = list(frames)
	segments = vad_collector(sample_rate, 30, 1000, vad, frames)
	
	for i, segment in enumerate(segments):
			a = []
			if segment is not None:
				path = os.path.join(folder,str(tracker)+'.wav')

				print(' Writing %s' % (path,))
				a.append(path)
				seg = segment[:200]+segment[previous_segment_length:] + segment[:200]
				write_wave(path, seg, sample_rate)
				tracker += 1
				previous_segment_length = len(segment)
				for audio in a:
					print(audio)
					w = wave.open(audio,'r')
					print(f'Recognising Audio {audio}')
					frames = w.getnframes()
					buffer = w.readframes(frames)
					data16 = np.frombuffer(buffer, dtype=np.int16)
					text = model.stt(data16)
					print('Recognises: {}'.format(text))
					result = sentiment_scores(text)
					print(result)
					emotion = model_emotion.predict(audio)
					print(emotion)
					filename = os.path.join(folder,'result.txt')
					print(filename)
					result_writer(audio,emotion,result,filename)
					if file:
						filename = os.path.join(folder,'result_{}.txt'.format(file.split('.')[0].split('_')[-1]))
						file_res = open(filename,'w').close()
						if float(file.split('.')[0].split('_')[-1]) in labels:
							filename = os.path.join(folder,'result_{}.txt'.format(file.split('.')[0].split('_')[-1]))
							result_writer(audio,emotion,result,filename,file=True)


