from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression

from mdoel_training.data_preparation import CVData, Parameter
from mdoel_training.model_input_and_output_classes import ModelInput
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogisticRegressionModel:
    def __init__(self, cv_data: CVData, target_col: str, parameters: List[Parameter] = None):
        self.cv_data = cv_data
        self.parameters = parameters or self.get_default_parameters()
        self.target_col = target_col

    def get_default_parameters(self):

        return [
            # Parameter(name='penalty', value='l2'),
            # Parameter(name='C', value=1.0),
            Parameter(name='solver', value='lbfgs'),
            Parameter(name='max_iter', value=500)
        ]

    def train_model(self, X_train, y_train):
        params = {}
        for param in self.parameters:
            params[param.name] = param.value
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        return model

    def train_cv(self):
        cv_scores = []
        for train, test in self.cv_data.splits:
            X_train, y_train = train.drop(columns=[self.target_col]), train[self.target_col]
            X_test, y_test = test.drop(columns=[self.target_col]), test[self.target_col]
            model = self.train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            print("y_pred", y_pred)
            print("y_train", np.unique(y_train))
            import pandas as pd
            conf_mat = pd.crosstab(y_test, y_pred)
            print("conf_mat")
            print(conf_mat)

            cv_scores.append(self.calculate_metrics(y_test, y_pred))
        return cv_scores

    def calculate_metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
