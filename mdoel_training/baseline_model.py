from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score

from mdoel_training.data_preparation import CVData
from mdoel_training.models.lgbm_class import score_lgbm


class BaselineModel:
    def __init__(self, baseline_value=None, percentile=None):
        self.current_baseline = baseline_value
        self.percentile = percentile

    def fit(self, y):
        if self.current_baseline is None:
            if self.percentile is None:
                self.current_baseline = y.mode()[0]
            else:
                self.current_baseline = y.quantile(self.percentile)
        elif self.percentile is not None:
            self.current_baseline = y.quantile(self.percentile)

    def transform(self, X):
        return pd.Series([self.current_baseline] * len(X))

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(y)
        return self.transform(X)


def train_baseline(X_train, y_train, baseline_value=None, percentile=None):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param baseline_num: A number to use as the baseline (mean, median, mode, or percentile)
    :param percentile: Percentile to use as the baseline if baseline_num is not provided
    :return: A trained baseline model
    """
    model = BaselineModel(baseline_value=baseline_value, percentile=percentile)
    model.fit(y_train)
    return model


def predict_baseline(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def baseline_with_outputs(cv_data: CVData, target_col: str, baseline_num=None, percentile=None, metric_funcs: List[callable] = None):
    results = []
    if not metric_funcs:
        metric_funcs = [accuracy_score, precision_recall_fscore_support, recall_score, f1_score]
    for i, (train, test) in enumerate(cv_data.splits):
        X_train, y_train = train.drop(target_col, axis=1), train[target_col]
        X_test, y_test = test.drop(target_col, axis=1), test[target_col]
        model = train_baseline(X_train, y_train, baseline_num, percentile)
        prediction = predict_baseline(model, X_test)
        prediction_train = predict_baseline(model, X_train)
        scores = score_lgbm(y_test, prediction, metric_funcs)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = score_lgbm(y_train, prediction_train, metric_funcs)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model

