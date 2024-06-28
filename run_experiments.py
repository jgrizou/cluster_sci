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
_DATASET_DIR = os.path.join(_EXP_DIR, 'datasets')

_RESULTS_DIR = os.path.join(_EXP_DIR, 'results')
file_tools.ensure_dir(_RESULTS_DIR)


import random
import numpy as np

def set_seed(seed, verbose=False):
    if verbose:
        print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)


dataset_paths = file_tools.sort_filepaths([f for f in file_tools.list_folders(_DATASET_DIR)])
method_names = ['DummyScoring_Mean', 'LinearRegression', 'Shuffle_LinearRegression', 'SVR', 'RandomForest', 'MLP']
# training_sizes = [9234, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]
training_sizes = [9234, 100]
eeg_names = ['EEG_Raw', 'EEG_Net']

if __name__ == '__main__':

    import argparse

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Print a number given as an argument.")
    parser.add_argument('-N', type=int, required=True, help="The number to print")

    # Parse the arguments
    args = parser.parse_args()


    thread_number = args.N
    counter = -1
    for dataset_path in dataset_paths:
            
        train_folder = os.path.join(dataset_path, 'train')
        train_filepaths = file_tools.sort_filepaths(file_tools.list_files(train_folder, '*.npz'))

        test_folder = os.path.join(dataset_path, 'test')
        test_filepaths = file_tools.sort_filepaths(file_tools.list_files(test_folder, '*.npz'))

        for eeg_name in eeg_names:
            for train_filename in train_filepaths:    
                for test_filename in test_filepaths:
                    for training_size in training_sizes:
                        for method_name in method_names:

                            counter += 1
                            if counter != thread_number:
                                continue
                            else:
                                print("Working on thread {}".format(thread_number))

                            result_folder = file_tools.change_refpath(dataset_path, _DATASET_DIR, _RESULTS_DIR)
                            file_tools.ensure_dir(result_folder)

                            eeg_folder = os.path.join(result_folder, eeg_name)
                            file_tools.ensure_dir(eeg_folder)
                
                            method_folder = os.path.join(eeg_folder, method_name)
                            file_tools.ensure_dir(method_folder)
                
                            result_per_traintest_folder = os.path.join(method_folder, file_tools.get_filebasename(train_filename), file_tools.get_filebasename(test_filename))  
                            file_tools.ensure_dir(result_per_traintest_folder)
                            results_filename = os.path.join(result_per_traintest_folder, '{}.json'.format(training_size))

                            if os.path.exists(results_filename):
                                print("Skipping {}".format(results_filename))
                                continue
                            else:
                                print("Working on {}".format(results_filename))  # Print the current iteration

                            # we generate a seed between 0 and 100, we will store it in results for reproducibility
                            seed = np.random.randint(1000) 
                            set_seed(seed) # setting seed here for rep  roducibility, we will store the selected_indexes too in results for reproducibility

                            test_data = np.load(test_filename)
                            test_faces = test_data['test_faces']
                            
                            train_data = np.load(train_filename)

                            # import here to not slow down the silly loop
                            import scorers
                            import scoring_tools
                            import saving_tools
                            import dataset_tools
                            if eeg_name == "EEG_Raw":
                                eeg_to_use = dataset_tools.eeg_raw
                            elif eeg_name == "EEG_Net":
                                eeg_to_use = dataset_tools.eeg_net
                            
                            selected_indexes, faces_observed, eeg_observed = dataset_tools.generate_subtraining_split(train_data['faces_observed'], eeg_to_use, training_size)
                            
                            # that is our way to have a shuffled baseline to check that the EEG do have an impact 
                            # and check we have not leaked information in any way in our code
                            if 'Shuffle' in method_name:
                                np.random.shuffle(eeg_observed)
                            
                            results = {}
                            results['seed'] = seed
                            results['eeg_name'] = eeg_name
                            results['method_name'] = method_name
                            results['training_size'] = training_size
                            results['selected_indexes'] = selected_indexes # can be used to check reproducibility is working ok
                            
                            # that is the big run soring all the test_faces based on the observations
                            scoring_function = scorers.method_to_function_mapping[method_name]
                            results['test_scorings'] = scoring_tools.eval_all_hyp(faces_observed, eeg_observed, test_faces, scoring_function)

                            # we store filenames for convenience when analysing results
                            results['dataset_path'] = dataset_path
                            results['target_filename'] = os.path.join(dataset_path, 'target', 'target.npz') # we don't use it but store it here for convenience when analysing results
                            results['train_filename'] = train_filename 
                            results['test_filename'] = test_filename
        
                            saving_tools.save_dict_to_json(results, results_filename)








