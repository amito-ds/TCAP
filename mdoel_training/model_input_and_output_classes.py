from typing import List

import pandas as pd

from mdoel_training.data_preparation import Parameter, CVData


class ModelResults:
    def __init__(self,
                 model_name: str,
                 model, results: pd.DataFrame,
                 parameters: List,
                 predictions: pd.Series,
                 cv_data: CVData):
        self.model_name = model_name
        self.model = model
        self.results = results
        self.parameters = parameters
        self.predictions = predictions
        self.cv_data = cv_data

    def aggregate_results(self):
        aggregate_df = self.results.groupby(['type']).mean()
        aggregate_df.reset_index(inplace=True)

        return aggregate_df

    def print(self):
        print("printing ModelResults")
        print("model name", self.model_name)
        print("model type", type(self.model))


class ModelInput:
    def __init__(self, cv_data: CVData, parameters: List[Parameter], target_col: str):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
