import pandas as pd
from nltk.corpus import brown

from classes import NLP
from cleaning_chapter import cleaning
from feature_analyzing.feature_correlation import FeatureAnalysis
from preprocessing.preprocessing import preprocess_text, get_stemmer
from util import get_stopwords

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.fe_main import get_embeddings
from mdoel_training.data_preparation import CVData

df2 = load_data_pirates().assign(target='pirates').sample(1000)
brown_sent = brown.sents(categories='news')[:1000]
brown_sent = [' '.join(x) for x in brown_sent]
df1 = pd.DataFrame({'text': brown_sent}).assign(target='news')
# df2 = load_data_king_arthur().assign(target='king')
df = pd.concat([df1, df2])
print("df shape", df.shape)
#
# # # Clean the text column
get_sw = get_stopwords()
df['text'] = df['text'].apply(lambda x: cleaning.clean_text(x,
                                                            remove_stopwords_flag=True,
                                                            stopwords=get_sw))
# # preprocess the text column
df['clean_text'] = df['text'].apply(lambda x:
                                    preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))

### text analyzer
# smart_text_analyzer.analyze_text(df)

### create embedding
train_embedding, test_embedding = get_embeddings(training_data=df, corex=True, tfidf=False, bow=False, corex_dim=100)

# Create a CVData object
# cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
# nlp = NLP(train_embedding, text_column='text', target_column='target')
#
# FeatureAnalysis(train_embedding, target_column='target').run()
# print("woww")
# best_model: ModelResults = ModelCycle(cv_data=cv_data, target_col='target').get_best_model()

# # # # # Re
# organized_results = organize_results(best_model.results)
# analyze_results(organized_results, best_model.parameters)
# analyze_model(best_model.model, cv_data, target_label='target')
