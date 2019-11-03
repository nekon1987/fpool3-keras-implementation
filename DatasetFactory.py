from LabeledDataEntry import LabeledDataEntry
from pathlib import Path
import random
import numpy as np

def load_wav_files_and_assign_their_labels(wav_files_directory, label_for_files_found_in_directory):
    labeled_data_entries = []

    for wav_file in wav_files_directory.iterdir():
        labeled_data_entry = LabeledDataEntry(wav_file, label_for_files_found_in_directory)
        labeled_data_entries.append(labeled_data_entry)

    return labeled_data_entries

def load_training_data_and_its_labels():
    training_data_positives = load_wav_files_and_assign_their_labels(
        Path("dataset/training/positive-identification"), 'training-positive-identification')
    training_data_negatives = load_wav_files_and_assign_their_labels(
        Path("dataset/training/negative-identification"), 'training-negative-identification')

    all_training_data = training_data_positives + training_data_negatives
    random.shuffle(all_training_data)

    x_data = list(map(lambda x: x.mfcc, all_training_data))
    y_labels = list(map(lambda x: x.label, all_training_data))
    x_data = np.reshape(x_data, [670, 40, 32, 1])  # why?

    return np.asarray(x_data),  np.array(y_labels)

def load_evaluation_data_and_its_labels():
    evaluation_data_positives = load_wav_files_and_assign_their_labels(
        Path("dataset/evaluation/positive-identification"), 'training-positive-identification')
    evaluation_data_negatives = load_wav_files_and_assign_their_labels(
        Path("dataset/evaluation/negative-identification"), 'training-negative-identification')

    all_evaluation_data = evaluation_data_positives + evaluation_data_negatives
    random.shuffle(all_evaluation_data)

    x_data = list(map(lambda x: x.mfcc, all_evaluation_data))
    y_labels = list(map(lambda x: x.label, all_evaluation_data))
    x_data = np.reshape(x_data, [36, 40, 32, 1])

    return np.asarray(x_data),  np.array(y_labels)

class DatasetFactory:
    pass


