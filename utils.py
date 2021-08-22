import librosa
import librosa.display
import IPython
import numpy as np
from IPython.display import Audio
from IPython.display import Image

"""
Define Features from Mel-Spectrogram
"""

sample_rate = 48000

sample_rate = 48000

def feature_melspectrogram(
    waveform,
    sample_rate,
    fft = 1024,
    winlen = 512,
    window='hamming',
    hop=256,
    mels=128,
    ):
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=fft,
        win_length=winlen,
        window=window,
        hop_length=hop,
        n_mels=mels,
        fmax=sample_rate/2)

    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

    return melspectrogram

def feature_mfcc(
    waveform,
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    mels=128
    ):
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        #hop_length=hop,
        n_mels=mels,
        fmax=sample_rate/2
        ) #nuevo: dos salidas
    return mfc_coefficients, np.mean(mfc_coefficients.T,axis=0)

def get_features(waveforms, features, samplerate):
    file_count = 0
    # nuevo
    mean_features = []
    for waveform in waveforms:
        # nuevo dos salidas
        mfccs, mean_mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        mean_features.append(mean_mfccs)
        file_count += 1
    return features, mean_features


def get_waveforms(file):
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    return waveform_homo

def dividiendo_audio(audio,sr,audio_input_dim = 3):
    L = len(audio)
    total_time = int(np.floor(L/sr))
    new_y = audio[:total_time*sr]
    tiempo_extra = total_time%audio_input_dim
    new_total_time = total_time - tiempo_extra
    final = new_y[:(new_total_time)*sr]
    audio_batch = np.reshape(final,(int(new_total_time/audio_input_dim),(sr*audio_input_dim)))
    return audio_batch

def mfcc_input_audio(audio_batches,sample_rate):
    mfcc_feature_per_audio = []
    mfcc_feature_per_audio_mean = []
    for i in range(audio_batches.shape[0]):
        audio = audio_batches[i]
        transformation, mean_t = feature_mfcc(audio,sample_rate)
        mfcc_feature_per_audio.append(transformation)
        # nuevo
        mfcc_feature_per_audio_mean.append(mean_t)
    return mfcc_feature_per_audio, mfcc_feature_per_audio_mean
