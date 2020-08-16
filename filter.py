import wave
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
from pydub.silence import split_on_silence
import pandas as pd
import random,pickle
import speech_recognition as sr

tracker = 0
timestamp = 0

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
modelObj = open('model_angry.pkl','rb')
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
			if num_voiced > 0.9 * ring_buffer.maxlen:
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
			if num_unvoiced > 0.9 * ring_buffer.maxlen:

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


def transcription_filter(text):
	pass

	''' This Function will be used to write up the results in filename'''
	
def result_writer(audio_file,emotion,result,filename,file=False):
		
		global timestamp
		if file:
			timestamp = 0
		file = open(filename,'a')
		n = 'Positive'
		if emotion ==  'neutral':
			if result == 'POSITIVE':
				n = 'Positive'
				print(n)
			elif result == 'NEGATIVE':
				n = 'Negative'
				print(n)
			else:
				n = 'Neutral'
				print(n)	
			
		elif emotion == 'happy':
			if result == 'POSITIVE':
				n = 'Positive'
				print(n)
			elif result == 'NEGATIVE':
				n = 'Negative'
				print(n)
			else:
				n = 'Positive'
				print(n)

		elif emotion ==  'angry':
			if result == 'POSITIVE':
				n = 'Negative'
				print(n)

			elif result == 'NEGATIVE':
				n = 'Negative'
				print(n)

			else:
				n = 'Negative'
				print(n)
		audio,sr = librosa.load(audio_file,sr=16000)
		length = librosa.get_duration(filename=audio_file)
		audio = audio
		print(len(audio))
		new_timestamp = length + timestamp
		a = np.linspace(timestamp,new_timestamp,len(audio))
		print('Writing timestamp files....')
		for i in range(len(audio)):
			file.write(f'{a[i]},{audio[i]},{n}\n')
		timestamp = new_timestamp
def t(t):
	global timestamp
	global tracker
	timestamp = t
	tracker = t
def main_file(file,labels=[0,1],folder='uploads',overall=False):
	song = AudioSegment.from_wav(file) 

	# open a file where we will concatenate 
	# and store the recognized text 
	fh = open("recognized.txt", "w+") 
		
	# split track where silence is 0.5 seconds 
	# or more and get chunks 
	chunks = split_on_silence(song, 
		# must be silent for at least 0.5 seconds 
		# or 500 ms. adjust this value based on user 
		# requirement. if the speaker stays silent for 
		# longer, increase this value. else, decrease it. 
		min_silence_len =2000, 

		# consider it silent if quieter than -16 dBFS 
		# adjust this per requirement 
		silence_thresh = song.dBFS-3
	) 

	# create a directory to store the audio chunks. 
	try: 
		os.mkdir('uploads') 
	except(FileExistsError): 
		pass

	# move into the directory to 
	# store the audio files. 


	i = 0
	# process each chunk 
	print('Entering')
	print(chunks)
	for chunk in chunks: 
		rec = ' '
		# Create 0.5 seconds silence chunk 
		chunk_silent = AudioSegment.silent(duration = 10) 

		# add 0.5 sec silence to beginning and 
		# end of audio chunk. This is done so that 
		# it doesn't seem abruptly sliced. 
		audio_chunk = chunk_silent + chunk + chunk_silent 

		# export audio chunk and save it in 
		# the current directory. 
		print("saving chunk{0}.wav".format(i)) 
		# specify the bitrate to be 192 k 
		audio_chunk.export("uploads/chunk{0}.wav".format(i), bitrate ='192k', format ="wav") 

		# the name of the newly created chunk 
		audio_file = 'uploads/chunk'+str(i)+'.wav'

		print("Processing chunk "+str(i)) 

		# get the name of the newly created chunk 
		# in the AUDIO_FILE variable for later use. 
		

		# create a speech recognition object 
		r = sr.Recognizer() 

		# recognize the chunk 
		with sr.AudioFile(audio_file) as source: 
			# remove this if it is not working 
			# correctly. 
			r.adjust_for_ambient_noise(source) 
			audio_listened = r.listen(source) 

			try: 
				# try converting it to text 
				rec += r.recognize_google(audio_listened,language='en-IN,en-US') 
				# write the output to the file. 
				fh.write(rec+". ") 

			# catch any errors. 
			except sr.UnknownValueError: 
				print("Could not understand audio") 

			except sr.RequestError as e: 
				print("Could not request results. check your internet connection") 

		i += 1
		text = rec
		
		try:
			emotion = model_emotion.predict(audio_file)
		except:
			emotion = 'neutral'
		print(text)
		result = sentiment_scores(text)
		print(result)
		if overall:
			filename = os.path.join(folder,'result.txt')
			result_writer(audio_file,emotion,result,filename)
		else:
			filename = os.path.join(folder,'result_{}.txt'.format(file.split('.')[0].split('_')[-1]))
			file_res = open(filename,'w').close()
			if float(file.split('.')[0].split('_')[-1]) in labels:
				filename = os.path.join(folder,'result_{}.txt'.format(file.split('.')[0].split('_')[-1]))
				result_writer(audio_file,emotion,result,filename,file=True)

previous_segment_length = 0
def main(audio=None,file=False,folder=None,sample_rate=16000,labels=[0,1]):
	global previous_segment_length
	global timestamp
	global tracker
	import speech_recognition as sr
	r = sr.Recognizer()
	vad = webrtcvad.Vad()
	frames = frame_generator(30, audio, sample_rate)
	frames = list(frames)
	segments = vad_collector(sample_rate, 30, 3000, vad, frames)
	
	for i, segment in enumerate(segments):
			a = []
			
			path = os.path.join(folder,str(tracker)+'.wav')

			print(' Writing %s' % (path,))
			a.append(path)
			
			seg = segment[:200]+segment[previous_segment_length-60000:]
			
			write_wave(path, seg, sample_rate)
			tracker += 1
			previous_segment_length = len(segment)
			for audio in a:
				
				with sr.AudioFile(audio) as source:
					# listen for the data (load audio to memory)
					r.adjust_for_ambient_noise(source)
					audio_data = r.record(source)
					# recognize (convert from speech to text)
					try:
						text = r.recognize_google(audio_data,language='en-IN,en-US')
					except:
						text = ''


					
				print('Recognises: {}'.format(text))
				result = sentiment_scores(text)
				print(result)
				try:
					emotion = model_emotion.predict(audio)
				except:
					emotion = 'neutral'
				print(emotion)
				filename = os.path.join(folder,'result.txt')
				print(filename)
				result_writer(audio,emotion,result,filename)

			
					


