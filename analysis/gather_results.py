import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding tools directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', 'tools')
sys.path.append(root_path)


import file_tools
import saving_tools
import numpy as np
import pandas as pd


_CSV_DIR = os.path.join('.', 'csv_files')
_SUBJECT_TO_TARGET_FILEPATH = os.path.join(_CSV_DIR, 'subject_to_target.csv')
_SUBJECT_TO_TARGET_DF = pd.read_csv(_SUBJECT_TO_TARGET_FILEPATH, index_col=0)
_ALL_TARGETS = _SUBJECT_TO_TARGET_DF.values

def make_values_unique(arr, epsilon=1e-6):
    # Create a copy of the input array to avoid modifying the original
    unique_arr = arr.astype(float)  # Convert to float to allow adding small values

    # Find the indices of duplicate values
    _, indices = np.unique(unique_arr, return_inverse=True)
    duplicates = np.where(np.bincount(indices) > 1)[0]

    # Add random values to duplicate elements
    for duplicate in duplicates:
        mask = indices == duplicate
        unique_arr[mask] += np.random.uniform(-epsilon, epsilon, size=mask.sum())

    assert len(np.unique(unique_arr)) == len(unique_arr), "The resulting array does not have all unique elements."
            
    return unique_arr

def ensure_local_path(cluster_path, local_refpath):
    cluster_refpath = "/wrk-vakka/users/carlosto/self-calibrating/experiments"
    if cluster_refpath in cluster_path:
        local_path = file_tools.change_refpath(cluster_path, cluster_refpath, local_refpath)
    else:
        local_path = cluster_path
    return local_path


def entry_from_result_filename(experiment_dir, result_filename):

    results = saving_tools.load_dict_from_json(result_filename)

    target_filename = ensure_local_path(results['target_filename'], experiment_dir)
    target_data = np.load(target_filename)
    target = target_data['target_face']

    test_filename = ensure_local_path(results['test_filename'], experiment_dir)
    test_data = np.load(test_filename)
    test_faces = test_data['test_faces']
    y_true = test_data['true_distances']
    
    raw_fit_times = np.array([np.sum(r['raw_scores']['fit_time']) for r in results['test_scorings']])     
    shuffle_fit_times = np.array([np.sum(r['shuffled_scores']['fit_time']) for r in results['test_scorings']])     
    raw_score_times = np.array([np.sum(r['raw_scores']['score_time']) for r in results['test_scorings']])     
    shuffle_score_times = np.array([np.sum(r['shuffled_scores']['score_time']) for r in results['test_scorings']])   
    total_fit_time = np.sum(raw_fit_times + shuffle_fit_times + raw_score_times + shuffle_score_times)

    # scores = np.array([r['mean_scores']['test_neg_root_mean_squared_error'] for r in results['test_scorings']]) # old scoring method, less humans interpreatble
    raw_scores =  np.array([-np.mean(r['raw_scores']['test_neg_root_mean_squared_error']) for r in results['test_scorings']])
    shuffled_scores = np.array([-np.mean(r['shuffled_scores']['test_neg_root_mean_squared_error']) for r in results['test_scorings']])
    
    scores = np.array([np.mean(np.array(r['shuffled_scores']['test_neg_root_mean_squared_error']) / np.array(r['raw_scores']['test_neg_root_mean_squared_error'])) for r in results['test_scorings']])
    scores = make_values_unique(scores)
    scores_sorted_indexes = np.flip(np.argsort(scores))

    true_sorted_indexes = np.argsort(y_true)

    infos = {}
    infos['method_name'] = [results['method_name']]

    if 'ablation_distance' in results:
        infos['ablation_distance'] = [results['ablation_distance']]

    if 'run_type' in results:
        infos['run_type'] = [results['run_type']]

    if 'n_component' in results:
        infos['n_component'] = results['n_component']

    if 'training_size' in results:
        infos['training_size'] = results['training_size']

    
    infos['eeg_name'] = results['eeg_name']
    infos['test_name'] = [file_tools.get_filebasename(results['test_filename'])]

    infos['fit_time'] = total_fit_time

    source_id_index = np.where((_ALL_TARGETS == target[None, :]).all(axis=1))[0][0]
    infos['source_id'] =  _SUBJECT_TO_TARGET_DF.iloc[source_id_index].name

    infos['raw_scores'] = [raw_scores]
    infos['shuffled_scores'] = [shuffled_scores]
    infos['target_raw_score'] = raw_scores[true_sorted_indexes][0]
    infos['target_shuffled_score'] = shuffled_scores[true_sorted_indexes][0]

    infos['scores'] = [scores]
    infos['true_distances'] = [y_true]

    if 'selected_indexes' in results:
        assert len(np.unique(results['selected_indexes'])) == len(results['selected_indexes']), "selected_indexes does not have all unique elements."

    infos['euclidean_at_top_rank'] = y_true[scores_sorted_indexes][0]
    
    from scipy import stats
    res = stats.pearsonr(y_true, scores, alternative='less')
    infos['pearsonr_statistic'] = res.statistic
    infos['pearsonr_pvalue'] = res.pvalue

    res = stats.spearmanr(y_true, scores, alternative='less')
    infos['spearmanr_statistic'] = res.statistic
    infos['spearmanr_pvalue'] = res.pvalue


    infos['target_rank'] = 1 + np.where(y_true[scores_sorted_indexes] == 0)[0][0] # we add 1 because index starts at 0 but rank at 1

    return pd.DataFrame.from_dict(infos)


if __name__ == '__main__':


    _EXP_DIR = os.path.join('..', 'experiments')
    _DF_DIR = os.path.join('.', 'df_files')

    file_tools.ensure_dir(_DF_DIR)

    all_result_folders = filter(lambda x: os.path.basename(x).startswith("result"), file_tools.list_folders(_EXP_DIR))

    for result_folder in all_result_folders:
        result_files = file_tools.list_files(result_folder, '*.json')
        
        df_filename = os.path.join(_DF_DIR, "{}.parquet".format(os.path.basename(result_folder)))
        if os.path.exists(df_filename):
            print("Skipping {}".format(result_folder))
            continue
        else:
            print("Working on {}".format(result_folder))

        df = pd.DataFrame()
        for i, result_filename in enumerate(result_files):
            print(f'Iteration {i+1}/{len(result_files)} for {result_folder}')  # Print the current iteration

            entry = entry_from_result_filename(_EXP_DIR, result_filename)
            df = pd.concat([df, entry], ignore_index=True)

        df.to_parquet(df_filename, engine='pyarrow')

