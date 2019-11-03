import librosa
import numpy as np
import os

class LabeledDataEntry(object):

    def __init__(self, file_path, label):
        self.file_path = file_path
        self.sample_rate = 16000 #read from file

        if label == 'training-negative-identification':
            self.label = 0
        if label == 'training-positive-identification':
            self.label = 1

        self.extract_audio_features()

    def extract_audio_features(self):
        audio, sample_rate = librosa.load(self.file_path, sr=self.sample_rate, res_type='kaiser_best')
        self.mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        self.mfccScaled = np.mean(self.mfcc.T, axis=0)
        self.remove_unsuitable_training_file()

    def remove_unsuitable_training_file(self):
        if self.mfcc.shape[1] != 32:
            os.remove(self.file_path)