from typing import List

import pandas as pd
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# path = os.path.abspath("TCAP")
# sys.path.append(path)
from cleaning_chapter import cleaning
from data_loader.webtext_data import load_data_pirates, load_data_chat_logs
from feature_analyzing.feature_correlation import FeatureAnalysis
from features_engineering import fe_main
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from mdoel_training.data_preparation import Parameter
from mdoel_training.model_input_and_output_classes import ModelResults
from mdoel_training.models.lgbm_class import lgbm_with_outputs, train_lgbm
from mdoel_training.models.lr import LogisticRegressionModel
from preprocessing import preprocessing
from text_analyzer.smart_text_analyzer import analyze_text


class NLP:
    def __init__(self,
                 df: pd.DataFrame,
                 text_column: str = 'text',
                 target_column: str = None,
                 train: pd.DataFrame = None,
                 test: pd.DataFrame = None,
                 cv_data: CVData = None,
                 model: ModelResults = None):
        self.df = df
        self.text_column = text_column
        self.target_column = target_column
        self.train = train
        self.test = test
        self.cv_data = cv_data
        self.model = model


class CleanText:
    def __init__(self,
                 nlp: NLP,
                 remove_punctuation_flag: bool = True,
                 remove_numbers_flag: bool = True,
                 remove_whitespace_flag: bool = True,
                 remove_empty_line_flag: bool = True,
                 lowercase_flag: bool = True,
                 remove_stopwords_flag: bool = False,
                 stopwords: List[str] = None,
                 remove_accented_characters_flag: bool = True,
                 remove_special_characters_flag: bool = True,
                 remove_html_tags_flag: bool = True):
        self.nlp = nlp
        self.remove_punctuation_flag = remove_punctuation_flag
        self.remove_numbers_flag = remove_numbers_flag
        self.remove_whitespace_flag = remove_whitespace_flag
        self.remove_empty_line_flag = remove_empty_line_flag
        self.lowercase_flag = lowercase_flag
        self.remove_stopwords_flag = remove_stopwords_flag
        self.stopwords = stopwords
        self.remove_accented_characters_flag = remove_accented_characters_flag
        self.remove_special_characters_flag = remove_special_characters_flag
        self.remove_html_tags_flag = remove_html_tags_flag


def get_default_clean_text(nlp: NLP) -> CleanText:
    return CleanText(nlp)


class Preprocessing:
    def __init__(self, nlp: NLP, stemmer=None, lemmatizer=None, stem_flag=True,
                 lemmatize_flag=False, tokenize_flag=True, pos_tag_flag=False):
        self.nlp = nlp
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.stem_flag = stem_flag
        self.lemmatize_flag = lemmatize_flag
        self.tokenize_flag = tokenize_flag
        self.pos_tag_flag = pos_tag_flag

    def print(self):
        print("lemmatize_flag", self.lemmatize_flag)
        print("nlp text col", self.nlp.text_column)


def get_default_preprocess_text(nlp: NLP) -> Preprocessing:
    return Preprocessing(nlp)


class TextAnalyzer:
    def __init__(self, nlp: NLP,
                 create_wordcloud: bool = True,
                 corex_topics: bool = True,
                 key_sentences: bool = True,
                 common_words: bool = True,
                 sentiment: bool = True,
                 data_quality: bool = True,
                 corex_topics_num: int = 10,
                 top_words: int = 10,
                 n_sentences: int = 5):
        self.nlp = nlp
        self.create_wordcloud = create_wordcloud
        self.corex_topics = corex_topics
        self.key_sentences = key_sentences
        self.common_words = common_words
        self.sentiment = sentiment
        self.data_quality = data_quality
        self.corex_topics_num = corex_topics_num
        self.top_words = top_words
        self.n_sentences = n_sentences

    def run(self):
        analyze_text(df=self.nlp.df,
                     text_column=self.nlp.text_column,
                     create_wordcloud=self.create_wordcloud,
                     corex_topics=self.corex_topics,
                     key_sentences=self.key_sentences,
                     common_words=self.common_words,
                     sentiment=self.sentiment,
                     data_quality=self.data_quality,
                     corex_topics_num=self.corex_topics_num,
                     top_words=self.top_words,
                     n_sentences=self.n_sentences)


def get_default_text_analyzer(nlp: NLP) -> TextAnalyzer:
    return TextAnalyzer(nlp)


class FeatureEngineering:
    def __init__(self, nlp: NLP,
                 training_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 split_data: bool = True,
                 split_prop: float = 0.3,
                 split_random_state=42,
                 text_column='text', target_col='target',
                 corex=True, corex_dim=100, tfidf=True, tfidf_dim=100, bow=True, bow_dim=100,
                 ngram_range=(1, 3)):
        self.nlp = nlp
        self.training_data = training_data
        self.test_data = test_data
        self.split_data = split_data
        self.split_prop = split_prop
        self.split_random_state = split_random_state
        self.text_column = text_column
        self.target_col = target_col
        self.corex = corex
        self.corex_dim = corex_dim
        self.tfidf = tfidf
        self.tfidf_dim = tfidf_dim
        self.bow = bow
        self.bow_dim = bow_dim
        self.ngram_range = ngram_range

    def convert_target_to_numerical(self) -> pd.DataFrame:
        df[self.target_col], _ = pd.factorize(self.nlp.df[self.target_col])
        return df

    def run_feature_engineering(self):
        # self.nlp.df = self.convert_target_to_numerical()
        if self.split_data:
            self.training_data, self.test_data = train_test_split(self.nlp.df, test_size=self.split_prop,
                                                                  random_state=self.split_random_state)
        return fe_main.get_embeddings(training_data=self.training_data,
                                      test_data=self.test_data,
                                      split_data=False,
                                      text_column=self.text_column,
                                      target_column=self.target_col,
                                      corex=self.corex, corex_dim=self.corex_dim, tfidf=self.tfidf,
                                      tfidf_dim=self.tfidf_dim,
                                      bow=self.bow, bow_dim=self.bow_dim, ngram_range=self.ngram_range)


def get_default_feature_engineering(nlp: NLP) -> FeatureEngineering:
    return FeatureEngineering(nlp)


class PreModelAnalysis:
    def __init__(self, nlp: NLP,
                 top_n_features: int = 200, correlation_matrix=True, tsne_plot=True,
                 top_n_pairplot=True,
                 chi_square_test_all_features=True):
        self.nlp = nlp
        self.top_n_features = top_n_features
        self.correlation_matrix = correlation_matrix
        self.tsne_plot = tsne_plot
        self.top_n_pairplot = top_n_pairplot
        self.chi_square_test_all_features = chi_square_test_all_features

    def run(self):
        FeatureAnalysis(df=self.nlp.train,
                        target_column=self.nlp.target_column,
                        top_n_features=self.top_n_features).run()


def get_default_pre_model_analysis(nlp: NLP):
    return PreModelAnalysis(nlp)


class ModelTraining:
    def __init__(self,
                 nlp: NLP,
                 parameters: List[Parameter] = None,
                 metric_funcs: List[callable] = None,
                 folds: int = 5):
        self.nlp = nlp
        self.parameters = parameters
        self.metric_funcs = metric_funcs
        self.folds = folds

    def get_model(self):
        if self.nlp.cv_data is not None:
            cv_data = self.nlp.cv_data
        else:
            # print("here as should be")
            # print(self.nlp.train[0:3])
            cv_data = CVData(train_data=self.nlp.train, folds=self.folds)

        return ModelCycle(cv_data=cv_data,
                          parameters=self.parameters,
                          metric_funcs=self.metric_funcs).get_best_model()


def get_default_model_training(nlp: NLP):
    return ModelTraining(nlp)


def clean_df_column(clean_text_obj: CleanText):
    df = clean_text_obj.nlp.df
    text_column = clean_text_obj.nlp.text_column
    df[text_column] = df[text_column].apply(lambda x: cleaning.clean_text(x,
                                                                          remove_punctuation_flag=clean_text_obj.remove_punctuation_flag,
                                                                          remove_numbers_flag=clean_text_obj.remove_numbers_flag,
                                                                          remove_whitespace_flag=clean_text_obj.remove_whitespace_flag,
                                                                          remove_empty_line_flag=clean_text_obj.remove_empty_line_flag,
                                                                          lowercase_flag=clean_text_obj.lowercase_flag,
                                                                          remove_stopwords_flag=clean_text_obj.remove_stopwords_flag,
                                                                          stopwords=clean_text_obj.stopwords,
                                                                          remove_accented_characters_flag=clean_text_obj.remove_accented_characters_flag,
                                                                          remove_special_characters_flag=clean_text_obj.remove_special_characters_flag,
                                                                          remove_html_tags_flag=clean_text_obj.remove_html_tags_flag))
    return df


def preprocess_df_column(preprocessing_obj: Preprocessing):
    df = preprocessing_obj.nlp.df
    text_column = preprocessing_obj.nlp.text_column
    df[text_column] = df[text_column].apply(lambda x: preprocessing.preprocess_text(x,
                                                                                    stemmer=preprocessing_obj.stemmer,
                                                                                    lemmatizer=preprocessing_obj.lemmatizer,
                                                                                    stem_flag=preprocessing_obj.stem_flag,
                                                                                    lemmatize_flag=preprocessing_obj.lemmatize_flag,
                                                                                    tokenize_flag=preprocessing_obj.tokenize_flag,
                                                                                    pos_tag_flag=preprocessing_obj.pos_tag_flag))
    return df


class TCAP:
    def __init__(self, nlp: NLP,
                 clean_text: CleanText = None,
                 preprocessing: Preprocessing = None,
                 text_stats: TextAnalyzer = None,
                 feature_engineering: FeatureEngineering = None,
                 pre_model_analysis: PreModelAnalysis = None,
                 model_training: ModelTraining = None,
                 cv_data: CVData = None):
        self.nlp = nlp
        self.clean_text = clean_text or get_default_clean_text(self.nlp)
        self.preprocessing = preprocessing or get_default_preprocess_text(self.nlp)
        self.text_stats = text_stats or get_default_text_analyzer(self.nlp)
        self.feature_engineering = feature_engineering or get_default_feature_engineering(self.nlp)
        self.model_training = model_training or get_default_model_training(self.nlp)
        self.pre_model_analysis = pre_model_analysis or get_default_pre_model_analysis(self.nlp)
        self.cv_data = cv_data
        self.df = nlp.df
        self.text_column = nlp.text_column
        self.target_column = nlp.target_column
        # self.clean_text =

    def run(self, is_clean_text: bool = True,
            is_preprocess_text: bool = True,
            is_text_stats: bool = False,
            is_feature_extraction: bool = True,
            is_feature_analysis: bool = True,
            is_train_model: bool = False,
            is_model_analysis: bool = True):
        if is_clean_text:
            self.df = clean_df_column(self.clean_text)
        if is_preprocess_text:
            self.df = preprocess_df_column(self.preprocessing)
        if is_text_stats:
            self.text_stats.run()
        if is_feature_extraction:
            self.nlp.train, self.nlp.test = self.feature_engineering.run_feature_engineering()
        if is_feature_analysis:
            self.pre_model_analysis.run()
        if is_train_model:
            self.nlp.model = self.model_training.get_model()
            self.nlp.cv_data = self.nlp.model.cv_data


print("start...")
# ### get data
# df1 = load_data_chat_logs().assign(target='chat_logs')
# print("logs:", df1.shape[0])
# df2 = load_data_pirates().assign(target='pirate')
# print("pirates:", df2.shape[0])
# df = pd.concat([df1, df2])
df2 = load_data_pirates().assign(target='pirates').sample(1000)

brown_sent = brown.sents(categories='reviews')[:5000]
brown_sent = [' '.join(x) for x in brown_sent]
df1 = pd.DataFrame({'text': brown_sent}).assign(target='reviews')
# df2 = load_data_king_arthur().assign(target='king')
df = pd.concat([df1, df2])

#
# ## TCAP
tcap = TCAP(NLP(df, target_column='target'))
tcap.run()
print("train", tcap.nlp.train.to_csv("embedding.csv"))
print("test", tcap.nlp.test.to_csv("test_embedding.csv"))
# nlp = tcap.nlp
# print(nlp.cv_data.splits[0][0])
# nlp =
# cv_data = tcap.nlp.cv_data

# cv_data = CVData(train_data=tcap.nlp.train, target_col=tcap.nlp.target_column)
# cv_scores = LogisticRegressionModel(cv_data, target_col='target').train_cv()
# print(cv_scores)
# print(tcap.df[0:10])
# model.predict(train.drop(columns=['target']))


# cv_data = tcap.cv_data
# cv_data.print()
# print("cv data length:", print(len(tcap.cv_data.splits)))

# print("df: ", tcap.df[0:10])
# print("text column: ", tcap.text_column)
# print("target column: ", tcap.target_column)
