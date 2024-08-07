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

_RESULTS_DIR = os.path.join(_EXP_DIR, 'optim')
file_tools.ensure_dir(_RESULTS_DIR)


import random
import numpy as np

def set_seed(seed, verbose=False):
    if verbose:
        print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)


# method_names = ['DummyScoring_Mean', 'LinearRegression', 'Shuffle_LinearRegression', 'SVR', 'RandomForest', 'MLP']
# training_sizes = [9234, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]
# method_names = ['DummyScoring_Mean', 'LinearRegression', 'Shuffle_LinearRegression']

method_names = ['LinearRegression', 'Shuffle_LinearRegression', 'DummyScoring_Mean']
eeg_names = ['EEG_Raw']

# n_components_face = [10, 512]
n_components_face = [10]
n_repeats = 1

_N_TRIALS = 2500 #for 10D need 3k to converge convincingly
_BOUNDS = 15

if __name__ == '__main__':

    dataset_paths = file_tools.sort_filepaths([f for f in file_tools.list_folders(_DATASET_DIR)])
    for dataset_path in dataset_paths:
            
        train_folder = os.path.join(dataset_path, 'train')
        train_filepaths = file_tools.sort_filepaths(file_tools.list_files(train_folder, '*.npz'))
            
        for i_repeat in range(n_repeats):
            for train_filename in train_filepaths:    
                for method_name in method_names:
                    for eeg_name in eeg_names:
                        for n_component_face in n_components_face:

                            result_folder = file_tools.change_refpath(dataset_path, _DATASET_DIR, _RESULTS_DIR)
                            file_tools.ensure_dir(result_folder)

                            eeg_folder = os.path.join(result_folder, eeg_name)
                            file_tools.ensure_dir(eeg_folder)
                
                            method_folder = os.path.join(eeg_folder, method_name)
                            file_tools.ensure_dir(method_folder)
                
                            result_per_traintest_folder = os.path.join(method_folder, file_tools.get_filebasename(train_filename), file_tools.generate_n_digit_name(n_component_face))  
                            file_tools.ensure_dir(result_per_traintest_folder)
                            
                            results_filename = os.path.join(result_per_traintest_folder, 'results_{}.json'.format(file_tools.generate_n_digit_name(i_repeat)))

                            if os.path.exists(results_filename):
                                print("Skipping {}".format(results_filename))
                                continue
                            else:
                                print("Working on {}".format(results_filename))

                            # we generate a seed between 0 and 100, we will store it in results for reproducibility
                            seed = np.random.randint(1000) 
                            set_seed(seed) # setting seed here for rep  roducibility, we will store the selected_indexes too in results for reproducibility
                            
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


                            from sklearn.decomposition import PCA
                            all_eeg = eeg_to_use
                            pca_eeg = PCA(n_components=20)
                            pca_eeg.fit(all_eeg)
                            eeg_observed = pca_eeg.transform(all_eeg)

                            all_faces = train_data['faces_observed']
                            pca_face = PCA(n_components=n_component_face)
                            pca_face.fit(all_faces)
                            faces_observed = pca_face.transform(all_faces)

                            # that is our way to have a shuffled baseline to check that the EEG do have an impact 
                            # and check we have not leaked information in any way in our code
                            if 'Shuffle' in method_name:
                                np.random.shuffle(eeg_observed)

                            scoring_function = scorers.method_to_function_mapping[method_name]

                            import optuna
                            import numpy as np
                            from optuna.samplers import CmaEsSampler
                            
                            def objective(trial):
                                # Generate the parameters
                                x = [trial.suggest_float(f'x{i}', -_BOUNDS, _BOUNDS) for i in range(n_component_face)]
                                results = scoring_tools.eval_hyp(faces_observed, eeg_observed, x, scoring_function)
                                score = np.mean(results['shuffled_scores']['test_neg_root_mean_squared_error']) / np.mean(np.array(results['raw_scores']['test_neg_root_mean_squared_error']))
                                return score

                            def custom_callback(study, trial):
                                print(f"Working on {results_filename}")
                                print(f"Trial {trial.number + 1}/{_N_TRIALS} finished with value: {trial.value}")
                                print(f"Best value so far: {study.best_value}")

                            sampler = CmaEsSampler()
                            study = optuna.create_study(direction='maximize', sampler=sampler)
                            study.optimize(objective, n_trials=_N_TRIALS, callbacks=[custom_callback])

                            df = study.trials_dataframe()
                            df_filename = os.path.join(result_per_traintest_folder, 'df_{}.parquet'.format(file_tools.generate_n_digit_name(i_repeat)))
                            df.to_parquet(df_filename, engine='pyarrow')

                            import pickle as pk
                            pca_eeg_filename = os.path.join(result_per_traintest_folder, 'pca_eeg_{}.pkl'.format(file_tools.generate_n_digit_name(i_repeat)))
                            pk.dump(pca_eeg, open(pca_eeg_filename,"wb"))

                            pca_face_filename = os.path.join(result_per_traintest_folder, 'pca_face_{}.pkl'.format(file_tools.generate_n_digit_name(i_repeat)))
                            pk.dump(pca_face, open(pca_face_filename,"wb"))
                            
                            results = {}
                            results['seed'] = seed
                            results['eeg_name'] = eeg_name
                            results['method_name'] = method_name
                            results['n_component_face'] = n_component_face
                            results['_N_TRIALS'] = _N_TRIALS
                            results['_BOUNDS'] = _BOUNDS

                            results['best_params'] = list(study.best_params.values())
                            results['best_value'] = study.best_value
                            best_face = pca_face.inverse_transform(list(study.best_params.values()))
                            results['best_face'] = list(best_face)

                            from scipy.spatial import distance
                            target_filename = os.path.join(dataset_path, 'target', 'target.npz')
                            target_data = np.load(target_filename)
                            target_face = target_data['target_face']
                            results['best_distance'] = distance.euclidean(best_face, target_face)

                            # we store filenames for convenience when analysing results
                            results['dataset_path'] = dataset_path
                            results['target_filename'] = target_filename
                            results['train_filename'] = train_filename 
                            results['df_filename'] = df_filename
                            results['pca_eeg_filename'] = pca_eeg_filename
                            results['pca_face_filename'] = pca_face_filename

                            saving_tools.save_dict_to_json(results, results_filename)
