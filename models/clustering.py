def kmeans(n_clusters:int=10):
  from sklearn.cluster import KMeans
  kmeans_kwargs = {
    "n_clusters": n_clusters,
    "init": "random",
    "max_iter": 300,
    "random_state": 42,
  }
  kmeans = KMeans(**kmeans_kwargs)

  return kmeans


def gmm(n_components:int=3):
  from sklearn.mixture import GaussianMixture
  gmm_kwargs = {
    'n_components': n_components, 
    'random_state': 42,
  }
  gmm = GaussianMixture(**gmm_kwargs)

  return gmm

import os
import pickle
from utils.common import create_output_directory


def birch():
  pass


def save_model(model, dir_, name):
  create_output_directory(dir_)
  o_filepath = os.path.join(dir_, f'{name}.pkl')
  pickle.dump(model, open(o_filepath, "wb"), protocol=4)


def clustering(features, type_:str='kmeans'):
  if len(features)==0:
    return 0
    
  cluster_method = {
    "kmeans": kmeans(),
    "gmm": gmm(),
  }

  model = cluster_method[type_]
  model.fit(features)


import numpy as np
from tqdm import tqdm

def run_cluster(total_features, images_info, dir_, name:str="kmeans", type_:str="kmeans"):
  # select model
  # clustering
  if 10000 < len(images_info):
    n, _ = divmod(len(images_info), 10000)
    split_list = np.array_split(images_info, n)
  else:
    split_list = np.array_split(images_info, 1)

  features_list = []
  for split in tqdm(split_list, desc="Feature Extracting..."):
    features = []
    for image in split:
      features.append(total_features[image["name"]])
    features_list.append(features)
  model = clustering(features_list, type_)
  # save model
  save_model(model, dir_, name)
  # predict
  for image in images_info:
    features.append(total_features[image["name"]])
  t_labels = model.predict(features)

  labels = dict()
  for idx, image in enumerate(images_info):
    labels[image["name"]] = t_labels[idx]

  return labels


import math
def create_hierarchy(input_dir:str, model_name:str, output_dir: str, level:int=8) -> list:
  # create heap
  heap = []
  # root node
  item = {"id": 0, "image_dir": f"{input_dir}", "model": f"{model_name}_root", "name": "root", "level": "root"}
  heap.append(item)
  for l in range(level):
    for i in range(int(math.pow(2, l+1))-1, int(math.pow(2, l+2)-1)):
      t_name = heap[int((i-1)//2)]["name"]
      if t_name=="root": t_name = "" 
      added = 'A' if i%2 == 1 else 'B'
      
      nodename=t_name+added
      
      item = {"id": i, 
              "image_dir": f"{output_dir}_{nodename}", 
              "model": f"{model_name}_{nodename}", 
              "name": nodename, 
              "level": l}
      heap.append(item)

  return heap