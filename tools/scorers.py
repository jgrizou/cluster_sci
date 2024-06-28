import scoring_tools

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def DummyScoring_Mean(eegs_observed, distances_hypothesis):
    estimator = DummyRegressor(strategy="mean")
    return scoring_tools.get_relative_scores(estimator, eegs_observed, distances_hypothesis)

def LinRegScoring(eegs_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), LinearRegression())
    return scoring_tools.get_relative_scores(estimator, eegs_observed, distances_hypothesis)

def SVRScoring(eegs_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), SVR())
    return scoring_tools.get_relative_scores(estimator, eegs_observed, distances_hypothesis)

def RandomForestScoring(eegs_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), RandomForestRegressor())
    return scoring_tools.get_relative_scores(estimator, eegs_observed, distances_hypothesis)

def MLPScoring(eegs_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50, 25)))
    return scoring_tools.get_relative_scores(estimator, eegs_observed, distances_hypothesis)

method_to_function_mapping = {}
method_to_function_mapping['DummyScoring_Mean'] = DummyScoring_Mean
method_to_function_mapping['LinearRegression'] = LinRegScoring
method_to_function_mapping['Shuffle_LinearRegression'] = LinRegScoring
method_to_function_mapping['SVR'] = SVRScoring
method_to_function_mapping['RandomForest'] = RandomForestScoring
method_to_function_mapping['MLP'] = MLPScoring