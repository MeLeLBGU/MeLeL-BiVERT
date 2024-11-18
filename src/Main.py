import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Imports
import stopwordsiso
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import logging

# BabelNet
import py_babelnet as pb
import simplemma

# Corpus imports
import nltk
from nltk.corpus import stopwords

# Model imports
from TranslationModel import TransModel
from WordAlignment import word_alignment, cosine_similarity, preprocess
from BabelNet import BabelNet, babelnet_distance

# Training 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error

plt.rcParams["figure.figsize"] = (20,40)

nltk.download('stopwords')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.info("Starting MT BiVert Exploration")

## DATA PREPARE FUNCTIONS ##

stop_words_en = set(stopwords.words("english"))
stop_words_zh = set(stopwordsiso.stopwords(["zh"]))

def get_stopwords(src_lang):
  if src_lang=='zh':
    return stop_words_zh
  return stop_words_en

def biVert_data(bn, model, source, back_target, technique, src_lang):
  # Do word alignment
  word_pairs = word_alignment(source=source, target=back_target, model=model, technique=technique)
  src_length = len(source)
  src_length = 1 if src_length==0 else src_length
  row = np.zeros(6)
  stop_words = get_stopwords(src_lang)

  print(word_pairs)
  for s, t in word_pairs:
    # Define type
    if s==t:
      print(f'{s}<-->{t}: Type SAME')
    else:
      print(f'{s}<-->{t}')
      if s == '' or (len(t)>1 and len(s)<2):  # Extra
        print(f'Type EXTRA')
        row[0] += 1/src_length
      elif t == '' or (len(t)<2 and len(s)>1): # Missing
        print(f'Type MISS')
        row[1] += 1/src_length
      elif t in stop_words and s in stop_words:
          print(f'Type STOPWORD')
          row[2] += 1/src_length
      elif src_lang == 'zh':
        if len(t)<4 and len(s)<4:
          print(f'Type SENSE')
          row[5] += babelnet_distance(bn, model, s, t)
        else:
          row[5] += cosine_similarity(s, t, model)
      elif (len(t)>1 and len(s)>1):
        # Check lemmatizer
        s_lemma = simplemma.text_lemmatizer(s, lang = src_lang.lower())[0].lower()
        t_lemma = simplemma.text_lemmatizer(t, lang = src_lang.lower())[0].lower()
        print(f'Words lemma: {s}-{s_lemma}, {t}-{t_lemma}')
        if s_lemma == t_lemma: # Inflection
          print(f'Type INFL')
          row[3] += cosine_similarity(s, t, model)
        elif s_lemma in t_lemma or t_lemma in s_lemma: # Derivation
          print(f'Type DERI')
          row[4] += cosine_similarity(s, t, model)
        else:  # Sense
          print(f'Type SENSE')
          row[5] += babelnet_distance(bn, model, s_lemma, t_lemma)
      else:
        print(f'{s}<-->{t} Not Identified')

  return row

def back_translation(model, s):
  """Back translate the sentence t to s"""
  batch_trg = model.tokenizer.encode(s, truncation=True, max_length=512, return_tensors = "pt").to(device)
  generated_ids = model.model.generate(batch_trg, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens = 512)
  trg_sentence = model.tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens = True)[0]
  return trg_sentence

def create_X(df):
  """Create updated df in batches"""
  df_final = pd.DataFrame(columns=['lp', 'src', 'mt', 'ref', 'score', 'raw', 'annotators', 'domain', 'trg'])
  batch_size = 500
  total_rows = len(df)
  num_batches = int(len(df)/batch_size + 1)

  for i in range(79,num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_rows)
    batch_df = df.iloc[start_idx:end_idx]
    # batch_df = batch_df.notna()

    for lg in batch_df['lp'].unique():
      df_lg = batch_df.loc[df['lp']==lg]
      src_lang = lg[:2]
      trg_lang = lg[-2:]
      marianFront = TransModel(src_lang, trg_lang)
      marianBack = TransModel(trg_lang, src_lang)
      df_lg['trg'] = df_lg.apply(lambda row: back_translation(marianBack, row["mt"]), axis=1)
      df_lg.to_csv(f'{i}-zh-2021-trans-mqm.csv')
      df_final = pd.concat([df_final, df_lg])

  return df_final

def expand_data():
  """Function for expanding data and adding target back translated text"""
  csv_files = ["2021-mqm.csv"]
  for file_path in csv_files:
    df = pd.read_csv(file_path, encoding='utf8')
    df.head()
    df = create_X(df)
    name = 'new_'+file_path
    df.to_csv(name)

def create_X_data(file_path, model):
  """Create training data"""
  df = pd.read_csv(file_path, encoding = 'utf-8')
  df_final = pd.DataFrame(columns=['lp', 'src', 'mt', 'ref', 'score', 'raw', 'annotators', 'domain', 'trg', 'EXTRA','MISS','STOPWORD','INFL','DERI','SENSE'])
  batch_size = 1000
  total_rows = len(df)
  num_batches = int(len(df)/batch_size + 1)

  for i in range(23,num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_rows)
    batch_df = df.iloc[start_idx:end_idx]
    batch_df[['EXTRA','MISS','STOPWORD','INFL','DERI','SENSE']] = batch_df.apply(lambda row: biVert_data(bn, model, preprocess(row['src'], 'zh'), preprocess(row['trg'], 'zh'), 'TOK', 'zh'), axis=1).apply(pd.Series)
    
    # Save in batch dfs
    batch_df.to_csv(f'{i}-{file_path}')
    df_final = pd.concat([df_final, batch_df])

  return df_final

# Sample data creation
bn = BabelNet('EN','EN',url='path_to_babelnet_server/bn')
model = TransModel('en','es', device)
df = create_X_data('en-es-2022-mqm-trans-pred.csv', model)
df.to_csv('en-es-2022-mqm-trans-pred-BIVERT.csv')


## TRAINING ##
def normalize_column_to_range(df, column_name, min_value, max_value):
  df[column_name] = ((df[column_name] - min_value) / (max_value - min_value)) * (100.0)
  return df

# Sample for MQM 2021-train, 2022-pred
data = pd.read_csv('en-ru-2021-mqm-babel-train.csv', encoding='utf-8', index_col='id')
data_pred = pd.read_csv('en-ru-2022-mqm-babel-pred.csv', encoding='utf-8', index_col='id')

data['score'] = data['score'].apply(lambda x: 0 if x<0 else x) 
data = normalize_column_to_range(data,"score",data['score'].min(), data['score'].max())
data_pred = normalize_column_to_range(data_pred,"score",data_pred['score'].min(), data_pred['score'].max())

# Select necessary columns
X_pred = data_pred.iloc[:,9:15]
y_pred = data_pred['score']
y = data['score']

data_train, data_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=42, shuffle = True)
X_train = data_train.iloc[:,9:15]
X_val = data_val.iloc[:,9:15]

best_gb_model = GradientBoostingRegressor(n_estimators=500, random_state=42, max_depth=7, learning_rate=0.1) 
best_gb_model.fit(X_train, y_train)

# Evaluate on the validation set
val_predictions = best_gb_model.predict(X_val)
# Evaluate on the test set
predictions = best_gb_model.predict(X_pred)

print(f"Validation MSE: {mean_squared_error(y_val, val_predictions):.2f}")
print(f"Test MSE: {mean_squared_error(y_pred, predictions):.2f}")

feature_importances = best_gb_model.feature_importances_
print("Feature Importances:", feature_importances)

# GET STATS
data_pred['pred'] = predictions
df = data_pred.groupby('system')[['score', 'pred']].mean()/10
print("Pearson Correlation", df['pred'].corr(df['score'], method='pearson'))