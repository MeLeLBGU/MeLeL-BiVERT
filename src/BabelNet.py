import py_babelnet as pb
from py_babelnet.calls import BabelnetAPI
import requests
import pickle
import json
import networkx as nx
from networkx.drawing.layout import bipartite_layout
from networkx import bipartite
import matplotlib.pyplot as plt
from WordAlignment import cosine_similarity

class BabelNet:

  def __init__(self, searchLang, targetLang, url):
    # Load Babalnet API
    self.bn = BabelnetAPI('KEYAPI')
    self.searchLang = searchLang
    self.targetLang = targetLang
    self.url = url
    self.reqNum = 0
    self.session = requests.Session()
    # Load Cache dictionary
    self.cache = self.__load_cache()

  # ------------------------------------ Cache Methods ------------------------------------ #
  def __del__(self):
    self.session.close()

  # Load cache from babel.pkl file in drive
  def __load_cache(self) -> dict:
    cache = dict()

    try:
      file = open("path/babel.pkl", "rb")
      cache = pickle.load(file)
      file.close()
    except:
      print("No file babel.pkl. Created file")
      self.__save_cache(cache)

    return cache

  # Save updated cache to babel.pkl file in drive
  def __save_cache(self, cache: dict):
    file = open("path/babel.pkl", "wb")
    pickle.dump(cache, file)
    file.close()

  # Finished with babelnet class, save cache to file
  def save(self):
    print("Number of calls: ", self.reqNum)
    self.__save_cache(self.cache)

  # ---------------------------------------- Word Senses Methods ---------------------------------------- #

  # Get senses for word
  def get_senses_of_word(self, word):
    if (__name__, word) in self.cache:
      senses = self.cache[(__name__, word)]
    else:
      if self.url != "":
        try:
          params = {'function':'get_senses','lemma':word, 'src':self.searchLang, 'trg':self.targetLang}
          response = self.session.get(self.url, params=params)
          if response.status_code == 200:
            senses = json.loads(response.text)
            self.cache[(__name__, word)] = senses
          else:
            senses = []
        except Exception as e:
          print('Error: ', e)
          print('Response: ', response)
      else:
        senses = self.bn.get_senses(lemma = word, searchLang = self.searchLang, targetLang = self.targetLang)

        # Add number of calls and add to cache dictionary
        self.reqNum += 1
        self.cache[(__name__, word)] = senses

    return [sense for sense in senses]

  # Get senses for word excluding position of word
  def get_senses_of_word_without_pos(self, word, pos):
    senses = self.get_senses_of_word(word)
    return [sense for sense in senses if sense.properties.pos != pos]

  # Get number of senses for word
  def get_senses_num(self, word):
    return len(self.get_senses_of_word(word))

  # ---------------------------------------- Sysnets Methods ---------------------------------------- #

  # Get synsets ids of word
  def get_synset_ids_word(self, word):
    if (__name__, word) in self.cache:
      synsets = self.cache[(__name__, word)]
    else:
      if self.url != "":
        try:
          params = {'function':'get_synset_ids', 'lemma':word, 'src':self.searchLang, 'trg':self.targetLang}
          response = self.session.get(self.url, params=params)
          if response.status_code == 200:
            synsets = json.loads(response.text)
            self.cache[(__name__, word)] = synsets
          else:
            synsets = []
        except Exception as e:
          print('Error: ', e)
          print('Response: ', response)
      else:
        synsets = self.bn.get_synset_ids(lemma = word, searchLang = self.searchLang, targetLang = self.targetLang)

        # Add number of calls and add to cache dictionary
        self.reqNum += 1
        self.cache[(__name__, word)] = synsets

    return synsets

  # Get number of babelnet Ids (synsets) for word
  def get_synsets_num(self, word):
    return len(self.get_synset_ids_word(word))

  # Get word synsets Ids and positions of words
  def get_word_bns(self, word):
    babelnet_ids = []
    bnid2pos = []
    synsetids = []

    synsetids = self.get_synset_ids_word(word)
    babelnet_ids = [synsetid["id"] for synsetid in synsetids]
    bnid2pos = {synset["id"]:synset["pos"] for synset in synsetids}

    return babelnet_ids, bnid2pos

  # Get specific synset information
  def get_synset_info(self, synset_id):
    if (__name__, synset_id) in self.cache:
      synset = self.cache[(__name__, synset_id)]
    else:
      if self.url != "":
        try:
          params = {'function':'get_synset', 'synsetId':synset_id, 'trg':self.targetLang}
          response = self.session.get(self.url, params=params)
          if response.status_code == 200:
            synset = json.loads(response.text)
            self.cache[(__name__, synset_id)] = synset
          else:
            synset = {}
        except Exception as e:
          print('Error: ', e)
          print('Response: ', response)
      else:
        synset = self.bn.get_synset(id = synset_id, targetLang = self.targetLang)

        # Add number of calls and add to cache dictionary
        self.reqNum += 1
        self.cache[(__name__, synset_id)] = synset

    return synset

  # Get full lemmas of (senses) in synset, only verb-noun. Input - synsetId
  def get_bn_lemmas_of_synset(self, synsetid, word):
    senses_list = self.get_synset_info(synsetid)

    try:
      senses_list = senses_list["senses"]
      word_capital = word[0].upper() + word[1:] 
      # lemmas = [sense["properties"]["fullLemma"] for sense in senses_list if (sense["properties"]["pos"]=='VERB' or sense["properties"]["pos"]=='NOUN') and
      #                                                                         word_capital not in sense["properties"]["fullLemma"] and
      #                                                                         sense["properties"]["language"]==self.targetLang]
      lemmas = [sense["properties"]["fullLemma"] for sense in senses_list if word_capital not in sense["properties"]["fullLemma"] and
                                                                              sense["properties"]["language"]==self.targetLang]
    except:
      print("Synset ID: ",synsetid , "has no senses for word '", word, "'.")
      lemmas = []

    return lemmas

  # Get all words from all synsets of word (not unique)
  def get_all_synsets_words(self, word):
    babelnet_ids, bnid2pos = self.get_word_bns(word)
    all_words = []
    for id in babelnet_ids:
      lemmas = self.get_bn_lemmas_of_synset(id, word)
      all_words += lemmas

    return all_words

  # ---------------------------------------- Edges Methods ---------------------------------------- #

  def get_edges_synset(self, id):
    """
  Get edges of synset
  Input - synsetId, lang
  Output - list
  """
    if (__name__, id) in self.cache:
      edges = self.cache[(__name__, id)]
    else:
      if self.url != "":
        params = {'function':'get_outgoing_edges', 'synsetId':id}
        response = self.session.get(self.url, params=params)
        if response.status_code == 200:
          edges = json.loads(response.text)
          self.cache[(__name__, id)] = edges
        else:
          edges = []
      else:
        edges = self.bn.get_outgoing_edges(id=id)

        # Add number of calls and add to cache dictionary
        self.reqNum += 1
        self.cache[(__name__, id)] = edges
    # print(edges)
    return edges

  def get_synset_of_type(self, synsetid, lang, relation):
    result = [edge["target"] for edge in self.get_edges_synset(id=synsetid)]
    print("synset_of_type: ", result)
                    # if ((edge["language"] == lang or edge["language"] == "MUL") and edge["pointer"]["relationGroup"] == relation and edge["target"] != synsetid)]
    return result

  def get_hypernyms(self, synsetid, lang):
    """  Get all hypernyms of synset
  Input - synsetId, lang
  Output - list
  """
    hypernyms =self.get_synset_of_type(synsetid, lang, "HYPERNYM")
    return hypernyms

  def get_hyponyms(self, synsetid, lang):
    """
      Get all hyponyms of synset
  Input - synsetId, lang
  Output - list
  """
    hyponyms = self.get_synset_of_type(synsetid, lang, "HYPONYM")
    return hyponyms

  def get_antonym(self, synsetid, lang):
    """
      Get all antonym of synset
  Input - synsetId, lang
  Output - list
  """
    antonym = self.get_synset_of_type(synsetid, lang, "ANTONYM")
    return antonym

  def get_other_relations(self, synsetid, lang):
    """  Get all other relations of synset
    Input - synsetId, lang
    Output - list
    """
    others = self.get_synset_of_type(synsetid, lang, "OTHER")
    return others
  

# GRAPH
# Using networkx Relationships:

# Grey: Synset Meaning
# Blue: Antonym
# Green: Hyponym
# Red: Hypernym
# Yellow: Other

def new_layer(bn, nodes, graph):
  src_lang = bn.searchLang
  nodes_to_add = []
  edges_to_add = []

  for synsetId in nodes:
    hypernyms = bn.get_hypernyms(synsetid=synsetId, lang=src_lang)
    nodes_to_add.extend((synsetId, {'lang': bn.targetLang, 'color': '#CBC3E3'}) for synsetId in hypernyms)
    edges_to_add.extend((synsetId, node, {
                'color': 'red',
                'key': 'hyper',
                'connectionstyle': 'arc3, rad=0',
                'weight': 1,
                'rad': 0.1,
            }) for node in hypernyms)

  graph.add_nodes_from(nodes_to_add, data=True)
  graph.add_edges_from(edges_to_add)
  if nodes_to_add:
    x_values, _ = zip(*nodes_to_add)
    x_values = list(x_values)
  else:
    x_values = []

  return x_values

def extend_graph(main_graph, bn, nodes, src_word, trg_word):
  # Get leaves of graph, last layer
  main_graph_depth = 1
  while not nx.has_path(main_graph, src_word, trg_word) and main_graph_depth < 8:
    print(f'Extending graph, level {main_graph_depth}')
    nodes = new_layer(bn, nodes = nodes, graph = main_graph)
    main_graph_depth += 1

# Create initial graph with words and their senses
def create_graph(bn, words) -> nx.MultiGraph:
  print("Creating Graph")
  graph = nx.MultiGraph()

  # Collect nodes and edges in lists
  nodes_to_add = []
  edges_to_add = []

  # Get translations for each word in list (from babel) and add to the respective lists
  for w in words:
    w = w.lower()
    print(w)
    graph.add_nodes_from([(w, {'lang': bn.searchLang, 'color': '#D3D3D3'})])
    synsetIds = set([row.get('id') for row in bn.get_synset_ids_word(word=w)])

    nodes_to_add.extend((synsetId, {'lang': bn.targetLang, 'color': '#CBC3E3'}) for synsetId in synsetIds)
    edges_to_add.extend((w, synsetId, {
                'color': 'gray',
                'key': 'babel',
                'connectionstyle': 'arc3, rad=0',
                'weight': 4,
                'rad': 0.1,
                'desc': "root"
            }) for synsetId in synsetIds)

  # Add nodes and edges in bulk
  graph.add_nodes_from(nodes_to_add, data=True)
  graph.add_edges_from(edges_to_add)

  if nodes_to_add:
    synsets, _ = zip(*nodes_to_add)
    synsets = list(synsets)
  else:
    synsets = []

  return graph, synsets

def print_graph(G, save, file, words):
  plt.figure(figsize=(10, 10), dpi=200)

  # pos=nx.spring_layout(G, scale=1)
  pos=nx.fruchterman_reingold_layout(G)
  # pos = nx.bipartite_layout(G, words)
  edges = G.edges(data=True)
  colors_edge = nx.get_edge_attributes(G,'color').values()
  colors_node = nx.get_node_attributes(G,'color').values()
  edge_styles = nx.get_edge_attributes(G,'connectionstyle').values()
  edge_weights = list(nx.get_edge_attributes(G,'weight').values())

  nx.draw(G, node_color = colors_node, edge_color = colors_edge, node_size=200, width = edge_weights, with_labels = True)
  # Graph(G, node_layout=pos, width = edge_weights, with_labels = True)

  if save:
    plt.savefig(file, format="PNG")
  plt.show()

# SCORE - SENSE RELATION
def num_same_arcs(graph, node, arc_type_r):
  """Number of same arc type from node"""
  return sum(1 if next(iter(graph.get_edge_data(e[0], e[1]).keys())) == arc_type_r else 0 for e in iter(graph.edges(node)))

# # Node depth
# def calc_node_depth(graph, node):
#   return min(nx.shortest_path_length(graph, source = node, target = r) for r in words)

def calc_edge_weight(graph, node_start, node_end):
  # Get edge type
  edge_data = graph.get_edge_data(node_start, node_end)
  edge_typeR = next(iter(edge_data.keys()))

  if edge_typeR == "anto":
    weight = 2.5
  elif edge_typeR == "babel":
    weight = 0
  else:
    maxR = 2
    minR = 1

    # Count number of same arcs for start node
    num_arc_start = num_same_arcs(graph, node_start, edge_typeR)
    num_arc_end = num_same_arcs(graph, node_end, edge_typeR)

    # Get number of same arcs
    weight_start = maxR - ((maxR - minR)/num_arc_start)
    weight_end = maxR - ((maxR - minR)/num_arc_end)

    weight = (weight_start + weight_end) /2.0

  return weight

def distance(model, graph, node_start, node_end):
  """ Path weight """
  if nx.has_path(graph, node_start, node_end):
    print("Calculating distance, has path")
    path = nx.shortest_path(graph, node_start, node_end)
    total_weight = 0

    for i in range(len(path)-1):
      total_weight += calc_edge_weight(graph, path[i], path[i+1])

    if total_weight == 0:
      return 0

    return 2*(0.5-1/total_weight)

  else:
    print("Calculating distance, no path - cosine")
    return cosine_similarity(node_start, node_end, model).item()
  
def babelnet_distance(bn, model, src_word, trg_word):
  main_graph, synsets = create_graph(bn, [src_word, trg_word])
  extend_graph(main_graph, bn, synsets, src_word, trg_word)
  print_graph(main_graph, True, "main.png", [src_word, trg_word])

  return distance(model, main_graph, src_word, trg_word)
