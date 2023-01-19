import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning_chapter.cleaning import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.fe_main import get_embeddings
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing import preprocess_text, get_stemmer
from util import get_stopwords

# classification
# buckets + chi square

# regression
# feautre corrleation to the label

# both
# feature matrix correlation
# anomaly detection

tsne_plot_message = """
The t-SNE plot is a visualization tool that is used to reduce the dimensionality of the data 
and visualize the relationships between the features and the target label in a 2D space. 
The plot shows the transformed data points, where each point represents a sample in the dataset 
and its color represents the target label. The plot can help you understand 
the relationships between the features and the target label in a more intuitive way.
"""

correlation_matrix_message = """
The correlation matrix is a tool used to measure the strength and direction of the linear relationship 
between different features in the dataset. The correlation coefficient ranges from -1 to 1, where a value 
of 1 indicates a perfect positive correlation, meaning that as one feature increases, the other feature 
also increases, a value of -1 indicates a perfect negative correlation, meaning that as one feature 
increases, the other feature decreases, and a value of 0 indicates no correlation between the features. 
It's important to note that a high correlation does not necessarily imply causality, it just indicates 
that the two features are related. In this report, we present the correlation matrix of the features 
in the dataset using a heatmap, where a darker color indicates a stronger correlation. It's worth noting 
that the correlation between the feature and target column (if provided) is also presented. In general, 
a correlation coefficient of 0.7 or higher is considered a strong correlation, a coefficient between 
0.3 and 0.7 is considered a moderate correlation, and a coefficient below 0.3 is considered a weak correlation. 
However, it's important to consider the context of the problem and the domain knowledge when interpreting 
the correlation matrix.
"""

pvalues_message = """
The p-values plot is a visualization tool that is used to determine the significance of each feature in the dataset.
The plot shows the distribution of p-values calculated using the chi-squared test for independence.
A low p-value (typically less than 0.05) indicates that there is a significant association between the feature and the target label.
Low p-values can potentially indicate a strong relationship, but not always, and vice versa.
It's important to also consider other factors and features when interpreting the results.
"""

pairplot_message = """
The pairplot helps to visualize the relationships between different features in the dataset. 
It shows scatter plots of sampled features and their distribution. 
This can help identify patterns or outliers in the data. 
"""


class FeatureAnalysis:
    def __init__(self, df: pd.DataFrame, target_column: str = None, top_n_features: int = 200):
        self.df = df
        self.target_column = target_column
        self.is_model = not (not target_column)
        if top_n_features:
            self.df = self.df[self.select_top_variance_features(top_n_features)]
        if not (not target_column):
            self.df[target_column] = df[target_column]

    def get_model_type(self, class_threshold: int = 2):
        is_classification, is_regression = (False, False)
        if not self.is_model:
            return is_classification, is_regression
        target_values = self.df[self.target_column]
        if target_values.dtype == object:
            is_classification = True
        elif target_values.nunique() <= class_threshold:
            is_classification, is_regression = (True, True)
        else:
            is_regression = True
        return is_classification, is_regression

    def correlation_matrix(self):
        print(correlation_matrix_message)
        corr = self.df.corr()
        plt.figure(figsize=(10, 8))
        plt.title("Feature correlation matrix")
        sns.heatmap(corr, annot=True)
        plt.show()
        return corr

    def tsne_plot(self, n_components=3, perplexity=30.0, n_iter=1000):
        print(tsne_plot_message)
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import LabelEncoder
        X = self.df.drop(columns=self.target_column)
        y = self.df[self.target_column]
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        X_tsne = tsne.fit_transform(X)
        le = LabelEncoder()
        y = le.fit_transform(y)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
        plt.title("t-SNE Plot of Features and Target Label")
        plt.show()

    def top_n_pairplot(self, N=4, trimming_left=0.05, trimming_right=0.05):
        print(pairplot_message)
        if trimming_left > 0:
            print(f"left trimming {100 * trimming_left}% ")
        if trimming_right > 0:
            print(f"right trimming {100 * trimming_right}% ")
        import seaborn as sns
        X = self.df.drop(columns=self.target_column)
        y = self.df[self.target_column]

        if y.dtype == 'object':
            kendall_corr = []
            for col in X.columns:
                kendall_corr.append(stats.kendalltau(X[col], y)[0])
            top_n_features = [X.columns[i] for i in np.argsort(kendall_corr)[-N:]]
            X_top_n = X[top_n_features]
            X_top_n[self.target_column] = y
            sns.pairplot(X_top_n, hue=self.target_column)
        else:
            corr = X.corrwith(y)
            top_n_features = corr.nlargest(N).index
            sns.pairplot(X[top_n_features])
        plt.show()

    def chi_square_test(self, feature):
        from scipy.stats import chi2_contingency
        X = self.df[[feature, self.target_column]]
        X = X.dropna()
        crosstab = pd.crosstab(X[feature], X[self.target_column])
        chi2, p, dof, expected = chi2_contingency(crosstab)
        return p

    def chi_square_test_all_features(self, k=3):
        pd.options.mode.chained_assignment = None
        from sklearn.cluster import KMeans
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        pvalues = {}
        for feature in self.df.columns:
            if feature != self.target_column:
                X = self.df[[feature]]
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)
                X[feature + "_cluster"] = kmeans.labels_
                crosstab = pd.crosstab(X[feature + "_cluster"], self.df[self.target_column])
                chi2, p, dof, expected = chi2_contingency(crosstab)
                pvalues[feature] = p
        return pvalues

    def plot_pvalues(self, threshold=20):
        print(pvalues_message)
        """
        Plots the histogram of p-values of chi-square test of independence between all features and the target column.
        Only features with at least threshold observations will be included in the plot.

        Parameters:
        threshold (int): Minimum number of observations required for a feature to be included in the plot. Default is 20.
        """
        if not self.is_model:
            pass
        pvalues = self.chi_square_test_all_features()
        if len(pvalues) < threshold:
            print(f"Number of features is less than {threshold}. Not enough data for histogram plot.")
            pass
        sns.histplot(list(pvalues.values()), bins=20)
        plt.xlabel("p-values")
        plt.ylabel("Frequency")
        plt.title("Histogram of p-values of Chi-Square Test of Independence (Feature X label)")
        plt.show()

    def feature_correlation(self):
        print("performing feature correlation")
        # get model type (classification, regression, both)
        # get if its a model

        # if not a model -> feature correlation (all features are numerics)
        # else:
        # if regression: plot correlation table, heatmap. last column is the target. choose the right correlation metric for this problem
        # if calssification:plot correlation table, heatmap. last column is the target. choose the right correlation metric for this problem

    def run(self,
            correlation_matrix=True,
            tsne_plot=True,
            top_n_pairplot=True,
            chi_square_test_all_features=True):
        # TODO remove comment
        # if correlation_matrix:
        #     self.correlation_matrix()
        if tsne_plot:
            if not (not self.target_column):
                self.tsne_plot()
        if top_n_pairplot:
            if not (not self.target_column):
                self.top_n_pairplot()
        if chi_square_test_all_features:
            self.plot_pvalues()

    def select_top_variance_features(self, n=200):
        variances = self.df.var()
        top_features = variances.sort_values(ascending=False).head(n).index
        return top_features
