from scipy.spatial import distance

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler


def eval_hyp(faces_observed, eegs_observed, face_hypothesis, scoring_function):
    import warnings
    warnings.filterwarnings("ignore")
    distances_hypothesis = [distance.euclidean(face, face_hypothesis) for face in faces_observed]
    scores = scoring_function(eegs_observed, distances_hypothesis)
    return scores


def eval_all_hyp(faces_observed, eegs_observed, face_hypothesis, scoring_function):
    return [eval_hyp(faces_observed, eegs_observed, face_hypothesis[i, :], scoring_function) for i in range(face_hypothesis.shape[0])]


def get_relative_scores(estimator, eegs_observed, distances_hypothesis):
    
    n_splits = 10
    test_size = 0.1
    random_state = np.random.randint(1000)
        
    scaler = StandardScaler()
    # Silly way to apply scaler to a list and have output as a list. Silly but just to keep working with list here and not have to change anything downstream.
    scaled_distances_hypothesis = scaler.fit_transform(np.array(distances_hypothesis).reshape(-1, 1)).flatten().tolist()

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    # scorings=['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    scorings=['neg_root_mean_squared_error']

    # raw scores
    raw_scores = cross_validate(estimator, eegs_observed, scaled_distances_hypothesis, cv=ss, scoring=scorings, n_jobs=-1)
    
    # shuffled eeg to use as baseline
    rng = np.random.RandomState(random_state)
    shuffled_eeg = eegs_observed[rng.permutation(len(eegs_observed))]
    shuffled_scores = cross_validate(estimator, shuffled_eeg, scaled_distances_hypothesis, cv=ss, scoring=scorings, n_jobs=-1)

    # packing it together
    results = {}
    results['raw_scores'] = raw_scores
    results['shuffled_scores'] = shuffled_scores
    
    results['scores'] = {}
    results['mean_scores'] = {}
    results['std_scores'] = {}
    for scoring_method in scorings:
        s = 'test_{}'.format(scoring_method)
        results['scores'][s] = raw_scores[s] - shuffled_scores[s]
        results['mean_scores'][s] = np.mean(results['scores'][s])
        results['std_scores'][s] = np.std(results['scores'][s])
    
    # results['distances_hypothesis'] = distances_hypothesis
    # results['scaled_distances_hypothesis'] = scaled_distances_hypothesis

    return results