"""
spectrogram calculation times
"""

# -*- coding: utf-8 -*-
import audioBasicIO as io
import ShortTermFeatures as sF
import numpy as np
import os
import librosa
import time
import sys

def get_spectrogram(path, win, step, method):
    """
    get_spectrogram() is a wrapper to
    pyAudioAnalysis.ShortTermFeatures.spectrogram() with a caching functionality

    :param path: path of the WAV file to analyze
    :param win: short-term window to be used in spectrogram calculation
    :param step: short-term step to be used in spectrogram calculation
    :return: spectrogram matrix, time array, freq array and sampling freq
    """
    fs, s = io.read_audio_file(path)
    if method == "pyaudioanalysis":
        spec_val, spec_time, spec_freq = sF.spectrogram(s, fs,
                                                        round(fs * win),
                                                        round(fs * step),
                                                        False, True)
    elif method == "librosa":
        s = np.double(s)
        s = s / (2.0 ** 15)

        spec_val = np.abs(librosa.stft(s, round(fs * win), round(fs * step)))
        spec_freq = [float((f + 1) * fs) / (round(fs * step))
                     for f in range(spec_val.shape[0])]
        spec_time = [float(t * round(fs * step)) / fs
                     for t in range(spec_val.shape[1])]
    elif method == "shorttermfeatures":
        sF.feature_extraction(s, fs, round(fs * win), round(fs * step))
        
t1 = time.time()
get_spectrogram("small_10.wav", 0.002, 0.002, "shorttermfeatures")
t2 = time.time()
print((t2-t1) / 10.0)
t1 = time.time()
get_spectrogram("small_20.wav", 0.002, 0.002, "shorttermfeatures")
t2 = time.time()
print((t2-t1) / 20.0)
t1 = time.time()
get_spectrogram("small_30.wav", 0.002, 0.002, "shorttermfeatures")
t2 = time.time()
print((t2-t1) / 30.0)
t1 = time.time()
get_spectrogram("small_50.wav", 0.002, 0.002, "shorttermfeatures")
t2 = time.time()
print((t2-t1) / 50.0)

