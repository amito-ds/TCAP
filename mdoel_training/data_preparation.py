# from preprocessing.preprocessing import preprocess_df_text
import os
import random
import sys
from typing import List


path = os.path.abspath("TCAP")
sys.path.append(path)

from sklearn.model_selection import KFold


class CVData:
    def __init__(self, train_data,
                 test_data=None,
                 folds=5,
                 target_col: str = None):
        self.train_data = train_data
        self.test_data = test_data
        self.folds = folds
        self.target_col = target_col
        self.splits = self.cv_preparation(train_data=self.train_data, test_data=self.test_data, k_fold=self.folds)
        self.encoded_folds = []

    @staticmethod
    def cv_preparation(train_data, test_data=None, k_fold=0):
        if k_fold == 0:
            return None
        elif k_fold > 0:
            kf = KFold(n_splits=k_fold)
            splits = []
            for train_indices, test_indices in kf.split(train_data):
                train = train_data.iloc[train_indices]
                test = train_data.iloc[test_indices]
                splits.append((train, test))
            return splits

    def print(self):
        # Print information about train and test data
        print(f"Train data shape: {self.train_data.shape}")
        if self.test_data is not None:
            print(f"Test data shape: {self.test_data.shape}")
        else:
            print("Test data not provided.")
        print(f"Number of splits: {len(self.splits)}")
        # Print information about first split
        if self.splits:
            print(f"First split train data shape: {self.train_data.iloc[self.splits[0][0]].shape}")
            print(f"First split test data shape: {self.train_data.iloc[self.splits[0][1]].shape}")


class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class ComplexParameter:

    def __init__(self, name: str, options: List):
        self.name = name
        self.options = options
        self.validate_options()

    def validate_options(self):
        # check if options is a valid type (int, float, tuple, list)
        pass

    def sample(self):
        if isinstance(self.options, int):
            return Parameter(self.name, random.randint(0, self.options))
        elif isinstance(self.options, float):
            return Parameter(self.name, random.uniform(0, self.options))
        elif isinstance(self.options, tuple):
            return Parameter(self.name, random.uniform(self.options[0], self.options[1]))
        elif isinstance(self.options, list):
            return Parameter(self.name, random.choice(self.options))


class ComplexParameterSet:
    def __init__(self, parameters: List[ComplexParameter]):
        self.parameters = parameters

    def sample(self):
        return [parameter.sample() for parameter in self.parameters]


def print_report(parameters: List[Parameter]):
    for param in parameters:
        print(f"{param.name}: {param.value}")
