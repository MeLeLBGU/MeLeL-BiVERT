import numpy as np
import re
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

# TOKENIZATION
## Tokenization help functions for technique type of aggregating tokens before alignment algorithm
def aggregate_embeddings(embeddings_list, agg):
  if agg == 'AVG':
    return np.mean(embeddings_list, axis=0)
  elif agg == 'MAX':
    return np.maximum.reduce(embeddings_list)
  elif agg == 'FIRST':
    return embeddings_list[0]

def combine_embeddings(tokens, embeddings, agg):
  """
  Combine embedding, 3 agg options: AVG, MAX, FIRST"""
  indicies = [i for i,v in enumerate(tokens) if v.startswith('▁') or v in ".,?!:"] + [len(tokens)]
  result = [[''.join(tokens[indicies[i]:indicies[i + 1]]).replace("▁", ""),
             aggregate_embeddings(embeddings[indicies[i]:indicies[i + 1]], agg)] for i in range(len(indicies) - 1)]
  tok, emb = zip(*result)
  return list(tok), list(emb)

# WORD ALIGNMENT ALGORITHM
## Inputs source sentence, back translated sentence, technique of token aggregation, model for word embeddings
def same_tokens(X, Y):
  """
  Get arrays of indicies from similarity matrix and decide on indicies pairs of same tokens
  """
  indicies = []
  for i in range(len(X)-1):
    if (X[i]==X[i+1]):
      indicies.append([i,i+1])
  indicies = np.unique(indicies)

  if not indicies.any():
    X_diff = []
    Y_diff = []
    X_same = X
    Y_same = Y
  else:
    X_diff = np.take(X, indicies)
    Y_diff = np.take(Y, indicies)
    X_same = np.delete(X, indicies)
    Y_same = np.delete(Y, indicies)

    # DIFF
    maxNum = Y_diff[0]
    indicies_to_remove = []
    X_keep = [X_diff[0]]
    for i in range(1,len(X_diff)-1):
      if (X_diff[i] in X_keep or Y_diff[i]<=maxNum):
        indicies_to_remove.append(i)
      else:
        X_keep.append(X_diff[i])
        maxNum = Y_diff[i]
    X_diff = np.delete(X_diff, indicies_to_remove)
    Y_diff = np.delete(Y_diff, indicies_to_remove)


  # SAME
  result = np.maximum.accumulate(Y_same)
  indices_to_delete = np.where(Y_same < result)[0]
  X_same = np.delete(X_same, indices_to_delete)
  Y_same = np.delete(Y_same, indices_to_delete)

  X_res = np.concatenate((X_same, X_diff)).astype(int)
  Y_res = np.concatenate((Y_same, Y_diff)).astype(int)
  return X_res, Y_res

def preprocess(sentence, src_lang):
  if src_lang == 'zh':
    return preprocess_chinese(sentence)
  return preprocess_en(sentence)

def preprocess_chinese(sentence):
  regex_pattern = re.compile(r'[\u4e00-\u9fa5]+')
  text = "".join(regex_pattern.findall(sentence))
  return text

def preprocess_en(sentence):
  sentence = sentence.lower()
  sentence= re.sub(r"n\'t", " not", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"\'s", " is", sentence)
  sentence = re.sub(r"\'d", " would", sentence)
  sentence = re.sub(r"\'ll", " will", sentence)
  sentence = re.sub(r"\'t", " not", sentence)
  sentence = re.sub(r"\'ve", " have", sentence)
  sentence = re.sub(r"\'m", " am", sentence)
  sentence = re.sub(r'\W+', ' ', sentence)
  sentence = sentence.replace("[^a-zA-Z0-9]", " ")
  return sentence


def word_alignment(source, target, technique, model, device):
  """
  Word Alignment. Input: source sentence, target sentence, technique: (TOK, AVG, MAX, FIRST)
  """
  combined_words = []

  src_tokens = model.tokenizer.tokenize(source) + ['▁<EOS>']
  trg_tokens = model.tokenizer.tokenize(target) + ['▁<EOS>']

  print("Src tokens: ", src_tokens)
  print("Trg tokens: ", trg_tokens)
  src_encoded = model.tokenizer.encode(source, return_tensors = "pt").to(device)
  trg_encoded = model.tokenizer.encode(target, return_tensors = "pt").to(device)
  src_embeddings = model.model.get_encoder().embed_tokens(src_encoded)[0].detach().cpu().numpy()
  trg_embeddings = model.model.get_encoder().embed_tokens(trg_encoded)[0].detach().cpu().numpy()

  if technique != 'TOK':
    src_tokens, src_embeddings = combine_embeddings(tokens = src_tokens, embeddings = src_embeddings, agg = technique)
    trg_tokens, trg_embeddings = combine_embeddings(tokens = trg_tokens, embeddings = trg_embeddings, agg = technique)

  print(f"Tokens Source: {src_tokens}\nTokens Target: {trg_tokens}")

  # Make trg embeddings at least as long as source - No match will be blank
  diff = len(src_embeddings) - len(trg_embeddings)
  if diff > 0:
    trg_embeddings = np.pad(trg_embeddings, ((0, diff), (0, 0)), mode='constant')
    trg_tokens += tuple(["▁"] * diff)
  elif diff < 0:
    src_embeddings = np.pad(src_embeddings, ((0, -diff), (0, 0)), mode='constant')
    src_tokens += tuple(["▁"] * -diff)
  sim_matrix = util.cos_sim(src_embeddings, trg_embeddings)

  ## Find equal words
  src_eq_idx = np.where(sim_matrix>0.999)[0]
  trg_eq_idx = np.where(sim_matrix>0.999)[1]
  src_eq_idx, trg_eq_idx = same_tokens(src_eq_idx, trg_eq_idx)

  # Remove corresponding rows and columns
  modified_matrix = np.delete(sim_matrix, src_eq_idx, axis=0)
  modified_matrix = np.delete(modified_matrix, trg_eq_idx, axis=1)

  # Update matching tokens list
  src_tokens = np.delete(src_tokens, src_eq_idx)
  trg_tokens = np.delete(trg_tokens, trg_eq_idx)

  modified_matrix = 1-modified_matrix
  print(modified_matrix)
  if len(modified_matrix)>0 and len(modified_matrix[0])>0:
    indexes_match = linear_sum_assignment(modified_matrix)

    # If by token combine to words:
    if technique == 'TOK':
      indicies = [i for i,v in enumerate(src_tokens) if v.startswith('▁')] + [len(src_tokens)-1]
      for i in range(len(indicies) - 1):
        combined_src = ''.join(src_tokens[indicies[i]:indicies[i + 1]])
        combined_trg = ''.join([trg_tokens[t] for s,t in zip(indexes_match[0], indexes_match[1]) if s>=indicies[i] and s<indicies[i + 1]])
        combined_words.append((combined_src.replace("▁", ""), combined_trg.replace("▁", "")))
    else:
      for s,t in zip(indexes_match[0], indexes_match[1]):
        if t>=len(trg_tokens):
          combined_words.append((src_tokens[s].replace("▁", ""), ""))
        else:
          combined_words.append((src_tokens[s].replace("▁", ""), trg_tokens[t].replace("▁", "")))

  return combined_words

def cosine_similarity(word1, word2, model, device):
  """Pair Identification and Scores
    For each word pair identify method to apply:

    Inflection
    Derivation
    Same
    BabelNet
    Missing
    Extra
    Stopwords"""
  word1 = model.tokenizer.encode(word1, return_tensors = "pt").to(device)
  word2 = model.tokenizer.encode(word2, return_tensors = "pt").to(device)
  word1_embeddings = model.model.get_encoder().embed_tokens(word1)[0].detach().cpu().numpy()[0]
  word2_embeddings = model.model.get_encoder().embed_tokens(word2)[0].detach().cpu().numpy()[0]
  return util.cos_sim(word1_embeddings, word2_embeddings)