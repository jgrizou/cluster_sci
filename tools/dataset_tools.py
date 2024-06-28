import os
import random
import numpy as np
from scipy.spatial import distance

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_DATA_FILE = os.path.join(HERE_PATH, "../data/all_data_sorted.npz")

data = np.load(_DATA_FILE)

target_faces = data['target_faces']
observed_faces = data['observed_faces']
eeg_raw = data['eeg_raw']
eeg_net = data['eeg_net']

unique_targets = np.unique(target_faces, axis=0)


def get_debiased_dataset(eeg_data):
    euclideans =  [distance.euclidean(row1, row2) for row1, row2 in zip(observed_faces, target_faces)]
    e = np.array(euclideans)
    n_points_per_bins = 145
    bins = np.arange(0, 47, 1.5)

    samples_ids = []
    for i in range(bins.shape[0]-1):
        indexes = np.where((e >= bins[i]) & (e < bins[i+1]))[0]
        samples_ids.extend(random.sample(list(indexes), 145))

    return list(np.array(euclideans)[samples_ids]), np.array(eeg_data)[samples_ids, :]


def get_unique_diagonal_shift(epsilon = 1e-6):

    shift = observed_faces - target_faces

    # Find unique rows and their counts with the margin of uniqueness
    unique_rows, inverse_indices = np.unique(shift, axis=0, return_inverse=True)
    counts = np.bincount(inverse_indices)

    # Merge rows that are within the margin of uniqueness
    for i in range(len(unique_rows)):
        for j in range(i+1, len(unique_rows)):
            if np.all(np.isclose(unique_rows[i], unique_rows[j], atol=epsilon)):
                counts[i] += counts[j]
                counts[j] = 0

    # Remove rows with zero counts
    unique_rows = unique_rows[counts != 0]
    counts = counts[counts != 0]

    return unique_rows


def generate_subtraining_split(faces_observed, eeg_observed, n_observation_to_keep):
    indexes = range(len(faces_observed))
    selected_indexes = np.random.choice(indexes, n_observation_to_keep, replace=False)
    return selected_indexes, np.copy(faces_observed)[selected_indexes, :], np.copy(eeg_observed)[selected_indexes, :]