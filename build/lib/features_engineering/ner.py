import pandas as pd
import spacy
from nltk.corpus import brown

from cleaning_chapter.cleaning import clean_text
from preprocessing.preprocessing import preprocess_text
from util import get_stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from model_loader.local_models import ner_model_path


# def load_ner_model(model_name):
#     # Load the specified NER model
#     nlp = spacy.load(model_name)
#
#     return nlp

def load_ner_model(path: str = ner_model_path):
    return spacy.load(path)


def classify_text(text, nlp=load_ner_model()):
    # Use the NER model to identify named entities in the text
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]

    return named_entities


def get_ner_bow_embedding(training_data: pd.DataFrame, class_col="class", embedding_size=100,
                          test: pd.DataFrame = None):
    bow_col_prefix = "ner_bow"
    # Join the named entities in each element of the class_col column into a single string
    named_entities_str = training_data[class_col].apply(lambda x: " ".join(x))

    # Vectorize the named entities using a BOW model with a specified number of features
    vectorizer = CountVectorizer(max_features=embedding_size)
    X = normalize(vectorizer.fit_transform(named_entities_str))

    feature_names = list(vectorizer.vocabulary_.keys())
    # Create a DataFrame for the training data
    train_embedding_df = pd.DataFrame(X.todense(), columns=[f"{bow_col_prefix}_{word}" for word in feature_names])

    if test is not None:
        # Join the named entities in each element of the class_col column in the test data into a single string
        test_named_entities_str = test[class_col].apply(lambda x: " ".join(x))

        # Vectorize the named entities in the test data using the same BOW model
        test_X = normalize(vectorizer.transform(test_named_entities_str))

        # Create a DataFrame for the test data
        test_embedding_df = pd.DataFrame(test_X.todense(),
                                         columns=[f"{bow_col_prefix}_{word}" for word in feature_names])

        return train_embedding_df, test_embedding_df
    else:
        return train_embedding_df


# Define a function to create an embedding for the named entities in text data

