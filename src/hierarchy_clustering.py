import os
import pickle
import math
import numpy as np
from tqdm import tqdm
from models.feature_extraction import get_image_features
from utils.common import get_files


def save_pkl(outdir:str, o_filename:str, model):
  print(f'[D]model save directory: {outdir}')
  if not os.path.isdir(outdir):
    print('[D]The directory is not present. Creating a new one..')
    os.makedirs(outdir)
  else:
    print('[D]The directory is present.')

  o_filepath = os.path.join(outdir, f'{o_filename}.pkl')
  pickle.dump(model, open(o_filepath, "wb"), protocol=4)
 

def kmeans(outdir:str, o_filename: str, samples, epochs, _n_clusters):
  n_clusters = list()
  if isinstance(_n_clusters, int):
    n_clusters.append(_n_clusters)
  else:
    n_clusters = _n_clusters
  ''' K-Means clustering
  '''
  print(f'[I] Input Dataset Size: {len(samples)}')
  kmeans_kwargs = {
    "init": "random",
    "max_iter": 300,
    "random_state": 42,
  }

  if 10000 < len(samples):
    n, _ = divmod(len(samples), 10000)
    split_list = np.array_split(samples, n)
  else:
    split_list = np.array_split(samples, 1)

  # Extract Features
  features_list = []
  for split in split_list:
    features = get_image_features(samples)
    features_list.append(features)

  # Get Model and learning
  for n_cluster in n_clusters:
    if len(split_list) == 1:
      from sklearn.cluster import KMeans
      kmeans = KMeans(n_clusters=n_cluster, **kmeans_kwargs)
      epochs_ = 1
    else:
      from sklearn.cluster import MiniBatchKMeans
      kmeans = MiniBatchKMeans(n_clusters=n_cluster, **kmeans_kwargs)
      epochs_ = epochs
      
    for _ in tqdm(range(epochs_), desc='K-Means Learning'):
      for features in features_list:
        if len(split_list) == 1:
          kmeans.fit(features)          
        else:
          kmeans.partial_fit(features) ## Partially fitting data in batches

    save_pkl(outdir, o_filename, kmeans)


def _predict(samples, clustering_model):
  if 10000 < len(samples):
    n, _ = divmod(len(samples), 10000)
    split_list = np.array_split(samples, n)
  else:
    split_list = np.array_split(samples, 1)

  labels = []

  for split in tqdm(split_list, desc="Prediction(by split set)"):
    print(f'[D] subset type: {type(split)}, length: {len(split)}')
    features = get_image_features(samples)
    t_labels = clustering_model.predict(features)
    labels.extend(t_labels)

  return labels


def predict(model_dir, outdir, o_filename, samples, left, right):
  model_filename = os.path.join(model_dir, o_filename+'.pkl')
  c_model = pickle.load(open(model_filename, "rb"))
  l = _predict(samples, c_model)
  save(l, samples, outdir, left, right)


def is_directory(dir_):
  if not os.path.isdir(dir_):
    os.makedirs(dir_)

    
def save(labels, samples, outdir, left, right):
  is_directory(left)
  is_directory(right)

  from shutil import copyfile
  for idx, l in enumerate(tqdm(labels, desc='Copy file into clustering')):
    f_name = f'{os.path.basename(samples[idx]).split(".")[0]}.{os.path.basename(samples[idx]).split(".")[1]}'
    if l == 0:
      copyfile(samples[idx], os.path.join(outdir, left, f_name))
    else:
      copyfile(samples[idx], os.path.join(outdir, right, f_name))

'''
rm -r /data/kwkim/aadw/results/h_test
rm -r /data/kwkim/aadw/pretrained/h_test

tree /data/kwkim/aadw/results/h_test
tree /data/kwkim/aadw/pretrained/h_test
'''
if __name__=="__main__":
  level = 3
  heap = []

  input_dir = '/data/kwkim/dataset/bladder/test_patches'
  output_dir = '/data/kwkim/aadw/results/h_test/cluster'
  model_dir = '/data/kwkim/aadw/pretrained/h_test/test_20220519N001'
  model_name = 'kmeans'

  for i in range(int(math.pow(2, level)-1)):
    if i==0:
      item = {"id": i, "image_dir": f"{input_dir}", "model": f"{model_name}_{i}"}
    else: 
      item = {"id": i, "image_dir": f"{output_dir}_{i}", "model": f"{model_name}_{i}"}

    heap.append(item)

  ''' heap
    왼쪽 자식의 인덱스 = (부모의 인덱스) * 2
    오른쪽 자식의 인덱스 = (부모의 인덱스) * 2 + 1
    부모의 인덱스 = (자식의 인덱스) / 2
  '''
  for i in range(int(math.pow(2, level-1)-1)):
    input_dir = heap[i]["image_dir"]
    pretraied_name = heap[i]["model"]
    images = get_files(input_dir)
    kmeans(model_dir, pretraied_name, images, 20, 2)

    left_image_dir = heap[i*2+1]["image_dir"]
    right_image_dir = heap[(i+1)*2]["image_dir"]
    predict(model_dir, left_image_dir, pretraied_name, images, left_image_dir, right_image_dir)

    ### debug
    # print(f'[D]====== current node {i} =====')
    # print(f'[D]\tinput directory: {input_dir}')
    # print(f'[D]\tmodel directory: {os.path.join(model_dir, pretraied_name)}')
    # print(f'[D]\tleft child directory: {left_image_dir}')
    # print(f'[D]\tleft child directory: {right_image_dir}')

