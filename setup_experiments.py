import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding tools directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, 'tools')
sys.path.append(root_path)

import file_tools
_EXP_DIR = os.path.join('.', 'experiments')
file_tools.ensure_dir(_EXP_DIR)

_DATASET_DIR = os.path.join(_EXP_DIR, 'datasets')
file_tools.ensure_dir(_DATASET_DIR)

import sys
import numpy as np
import dataset_tools
import sampling_tools
from scipy.spatial import distance


def save_test_data(test_dict, test_folder_name, test_name):
    file_tools.ensure_dir(test_folder_name)
    
    assert 'X' in test_dict and 'y_true' in test_dict, "The dictionary must have 'X' and 'y_true' fields"
    assert len(test_dict['X']) == len(test_dict['y_true']), "The 'X' and 'y_true' fields must have the same size"
    
    test_filename = os.path.join(test_folder_name, '{}.npz'.format(test_name))
    np.savez_compressed(test_filename, test_faces=test_dict['X'], true_distances=test_dict['y_true'])


if __name__ == '__main__':

    distances_to_use = [distance.euclidean(row1, row2) for row1, row2 in zip(dataset_tools.observed_faces, dataset_tools.target_faces)]
    shift = dataset_tools.get_unique_diagonal_shift()
    start_faces = dataset_tools.unique_targets

    for i, start_face in enumerate(start_faces):
        folder_name = os.path.join(_DATASET_DIR, file_tools.generate_n_digit_name(i))
        file_tools.ensure_dir(folder_name)
        
        sys.stdout.write(f'\rIteration {i+1}')  # Print the current iteration
        sys.stdout.flush()

        # TARGET
        target_folder_name = os.path.join(folder_name, 'target')
        file_tools.ensure_dir(target_folder_name)
                
        target_filename = os.path.join(target_folder_name, 'target.npz')
        np.savez_compressed(target_filename, target_face=start_face)
                                            
        # TRAIN
        train_folder_name = os.path.join(folder_name, 'train')
        file_tools.ensure_dir(train_folder_name)
        
        # generate random observed datasets
        N_train_set = 10
        for i in range(N_train_set):
            train_filename = os.path.join(train_folder_name, '{}.npz'.format(file_tools.generate_n_digit_name(i)))
            if os.path.exists(train_filename):
                print("Skipping {} as already exist".format(train_filename))
                continue
    
            new_faces, _ = sampling_tools.optimise_latent_dataset_at_euclideans(start_face, distances_to_use)

            # no cheating, we do not even store the distance values so we cannot access them
            np.savez_compressed(train_filename, faces_observed=np.array(new_faces))
            
        # TEST
        test_folder_name = os.path.join(folder_name, 'test')
        file_tools.ensure_dir(test_folder_name)
        
        # generate diagonal test dataset and faces
        diagonal_faces = start_face + shift
        diagonal_test_faces = {}
        diagonal_test_faces['X'] = np.array(diagonal_faces)
        diagonal_test_faces['y_true'] = np.array([distance.euclidean(start_face, diag_face) for diag_face in diagonal_faces])
        
        assert start_face in diagonal_faces, "The test set must have the exact target face"

        save_test_data(diagonal_test_faces, test_folder_name, 'diagonal')
            
        # generate random test dataset and faces
        N_random_faces = diagonal_faces.shape[0] - 1 #-1 because we enforce to have the target face in the test set which we add after
        random_distances = list(np.random.rand(N_random_faces)*np.max(distances_to_use))
        random_faces, _ = sampling_tools.optimise_latent_dataset_at_euclideans(start_face, random_distances)
        random_faces.append(start_face)
        random_faces = np.array(random_faces)
        
        assert start_face in random_faces, "The test set must have the exact target face"
            
        random_test_faces = {}
        random_test_faces['X'] = np.array(random_faces)
        random_test_faces['y_true'] = np.array([distance.euclidean(start_face, rnd_face) for rnd_face in random_faces])

        save_test_data(random_test_faces, test_folder_name, 'random')
