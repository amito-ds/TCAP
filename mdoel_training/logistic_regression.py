import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

from mdoel_training.data_preparation import CVData
from typing import List
from sklearn.preprocessing import LabelBinarizer

import pandas as pd


def logistic_regression_with_outputs(cv_data: CVData, parameters: list, target_col: str,
                                     metric_funcs: List[callable] = None):
    results = []

    if not metric_funcs:
        metric_funcs = [accuracy_score, precision_recall_fscore_support, recall_score, f1_score]
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        model = LogisticRegression()
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        X_train = X_train.drop(columns=[target_col])
        X_test = X_test.drop(columns=[target_col])
        for param in parameters:
            setattr(model, param.name, param.value)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        # label handeling
        label_binarizer = LabelBinarizer()
        y_train = label_binarizer.fit_transform(y_train)
        y_test = label_binarizer.transform(y_test)
        train_pred = label_binarizer.transform(train_pred)
        test_pred = label_binarizer.transform(test_pred)

        train_scores = {metric.__name__: metric(y_train, train_pred) for metric in metric_funcs}
        test_scores = {metric.__name__: metric(y_test, test_pred) for metric in metric_funcs}
        results.append(
            {'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
        results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})
    model.fit(cv_data.train_data.drop(columns=[target_col]), cv_data.train_data[target_col])
    return results, model
