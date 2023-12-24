import sys,os
from pathlib import Path
sys.path.append('./custom_utils')

from custom_utils.datastft import single_spectrogram
import numpy as np
import matplotlib.pyplot as plt

import cv2
import math
from scipy.io import wavfile

import pylab
import json
import sys
import pickle


def _get_stft_spectogram(wav_raw, audio_rate):

    slope_wav = 0.0144
    intersec_wav = 0.8280000000000001
    freq = audio_rate
    wlen = 640
    hop = int(wlen/2)
 
    stft_data = single_spectrogram(wav_raw, freq, wlen, hop) * slope_wav + intersec_wav
    return np.swapaxes(stft_data, 0, 1).tolist()
