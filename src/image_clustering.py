'''Usage

[[=====]]
python src/image_clustering.py --mode predict --type kmeans \
--input_dir '/data/kwkim/dataset/bladder/patches-1024' \
--output_dir '/data/kwkim/aadw/results/gmm_6_1024/clustering' \
--model_dir '/data/kwkim/aadw/pretrained/gmm/gmm_6_size_1024_20220502' \
--trained 'gmm_6_size_1024_20220502' --n_cluster 4 


python src/image_clustering.py --mode train_and_predict --type kmeans \
--input_dir '/data/kwkim/dataset/bladder/test_patches' \
--output_dir '/data/kwkim/aadw/results/test/cluster' \
--model_dir '/data/kwkim/aadw/pretrained/test' \
--trained 'bladder_test' --n_cluster 3


python src/image_clustering.py --mode train_and_predict --type kmeans \
--input_dir '/data/kwkim/dataset/bladder/trainset_v2.1' \
--output_dir '/data/kwkim/aadw/results/bladder_kmeans_v2.1/20220518N002/cluster' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans/bladder_kmeans_v2.1_20220518N002' \
--trained 'bladder_kmeans_v2.1_20220518N002' --n_cluster 48 80 99 --epochs 20

<notes>
bladder_kmeans_v2.1_20220518N001: vgg16
bladder_kmeans_v2.1_20220518N002: resnet

python src/image_clustering.py --mode train_and_predict --type kmeans \
--input_dir '/data/kwkim/dataset/bladder/trainset_v1.0' \
--output_dir '/data/kwkim/aadw/results/bladder_kmeans_v1.0/20220518N001/cluster' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans/bladder_kmeans_v1.0_20220518N001' \
--trained 'bladder_kmeans_v1.0_20220518N001' --n_cluster 48 80 99 --epochs 20
<notes>
bladder_kmeans_v1.0_20220518N001: resnet


[[ GMM ]]
python src/image_clustering.py --mode debug \
--input_dir '/data/kwkim/aadw/dataset/bladder/patches' \
--output_dir '/data/kwkim/aadw/results/gmm_clustering' \
--model_dir '/data/kwkim/aadw/pretrained/gmm/20220412_bladder_trained' \
--trained 'bladder-gmm' --n_cluster 32 64

python src/image_clustering.py --mode predict \
--input_dir '/data/kwkim/aadw/dataset/bladder/patches_test' \
--output_dir '/data/kwkim/aadw/results/gmm_clustering' \
--model_dir '/data/kwkim/aadw/pretrained/gmm/20220412_bladder_trained' \
--trained 'bladder-gmm' --n_cluster 16 32 64

'''
''' Usage (Test server)
python src/image_clustering.py --mode train \
--input_dir '/home/aidev/Documents/kwkim/aadw/dataset/bladder/test_patches' \
--output_dir '/home/aidev/Documents/kwkim/aadw/results/test_clustering' \
--model_dir '/home/aidev/Documents/kwkim/aadw/test_trained' \
--trained 'kmeans-bladder-test' --n_cluster 16 32 64

python src/image_clustering.py --mode predict \
--input_dir '/home/aidev/Documents/kwkim/aadw/dataset/bladder/patches_test' \
--output_dir '/home/aidev/Documents/kwkim/aadw/results/gmm_clustering' \
--model_dir '/home/aidev/Documents/kwkim/aadw/pretrained/gmm/20220412_bladder_trained' \
--trained 'bladder-gmm' --n_cluster 16 32 64

python src/image_clustering.py --mode predict \
--input_dir '/home/aidev/Documents/kwkim/aadw/dataset/bladder/patches_test' \
--output_dir '/home/aidev/Documents/kwkim/aadw/results/kmeans_clustering' \
--model_dir '/home/aidev/Documents/kwkim/aadw/pretrained/kmeans/20220413' \
--trained 'bladder-kmeans' --n_cluster 16 32 64


'''
from tqdm import tqdm
import numpy as np

import os
import pickle
import argparse
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
 

def split_samples(list_:list) -> list:
  if 10000 < len(list_):
    n, _ = divmod(len(list_), 10000)
    split_list = np.array_split(list_, n)
  else:
    split_list = np.array_split(list_, 1)
  return split_list


def dbscan(outdir:str, o_filename: str, samples, n_clusters):
  print(f'[I] Input Dataset Size: {len(samples)}')
  features = get_image_features(samples)
  # DBSCAN
  from sklearn.cluster import DBSCAN
  # n_components로 미리 군집 개수 설정
  dbscan_kwargs = {
    'eps': 0.5,
    'min_samples': 12
  }
  print(f'[I] ========== GaussianMixture ==========')
  for n_cluster in n_clusters:
    dbscan = DBSCAN(**dbscan_kwargs)
    dbscan.fit(features)
    save_pkl(outdir, o_filename+'_'+str(n_cluster), dbscan)


def gmm(outdir:str, o_filename: str, samples, n_clusters):
  print(f'[I] Input Dataset Size: {len(samples)}')
  features = get_image_features(samples)
  # GMM 적용
  from sklearn.mixture import GaussianMixture
  # n_components로 미리 군집 개수 설정
  gmm_kwargs = {
    # 'n_components': 3, 
    'random_state': 42,
  }
  print(f'[I] ========== GaussianMixture ==========')
  for n_cluster in n_clusters:
    gmm = GaussianMixture(n_components=n_cluster, **gmm_kwargs)
    gmm.fit(features)
    save_pkl(outdir, o_filename+'_'+str(n_cluster), gmm)


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

    save_pkl(outdir, o_filename+'_'+str(n_cluster), kmeans)


def train(type_: str, outdir:str, o_filename: str, samples, epochs, n_clusters):
  if "kmeans"==type_:
    kmeans(outdir, o_filename, samples, epochs, n_clusters)
  elif "gmm"==type_:
    gmm(outdir, o_filename, samples, n_clusters)
  elif "dbscan"==type_:
    dbscan(outdir, o_filename, samples, n_clusters)


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


def predict(model_dir, outdir, o_filename, samples, _n_clusters):
# def predict(args, s):
  n_clusters = list()
  if isinstance(_n_clusters, int):
    n_clusters.append(_n_clusters)
  else:
    n_clusters = _n_clusters

  for n_cluster in n_clusters:
    model_filename = os.path.join(model_dir, o_filename+'_'+str(n_cluster)+'.pkl')
    c_model = pickle.load(open(model_filename, "rb"))
    l = _predict(samples, c_model)
    save(outdir+'_'+str(n_cluster), l, samples)
    

def save(outdir, labels, samples):
  max_label = max(labels)
  print(f'[D]output diredtory: {outdir}')
  if not os.path.isdir(outdir):
    print('[D]The directory is not present. Creating a new one..')
    os.makedirs(outdir)
  else:
    print('[D]The directory is present.')
  
  # make output sub-directory
  for sub_idx in range(max_label+1):
    try:
      os.makedirs(os.path.join(outdir, str(sub_idx)))
    except:
      continue

  from shutil import copyfile
  for idx, l in enumerate(tqdm(labels, desc='Copy file into clustering')):
    f_name = f'{os.path.basename(samples[idx]).split(".")[0]}.{os.path.basename(samples[idx]).split(".")[1]}'
    copyfile(samples[idx],
      os.path.join(outdir, str(l), f_name))


def main():
  parser = argparse.ArgumentParser(description='Image clustering')
  parser.add_argument('--mode', dest='mode', choices=['train', 'predict', 'train_and_predict'],
                      default='train-and-predict', 
                      help='Operation mode [train|predict|train_and_predict] (default: predict)')
  parser.add_argument('--type', dest='type', choices=['kmeans', 'gmm', 'dbscan'],
                      default='kmeans', 
                      help='Clustering model type')
  parser.add_argument('--input_dir', dest='indir', 
                      default='/data/kwkim/aadw/dataset/bladder/test_patches',
                      help='Train Dataset Directory (default: /data/kwkim/aadw/dataset/in)')
  parser.add_argument('--output_dir', dest='outdir', 
                      default='/data/kwkim/aadw/results/test_clustering',
                      help='Result Directory (default: /data/kwkim/aadw/dataset/out)')
  parser.add_argument('--model_dir', dest='mdir', 
                      default='/data/kwkim/aadw/test_trained',
                      help='Result Directory (default: /data/kwkim/aadw/dataset/out)')
  parser.add_argument('--trained', dest='trained', 
                      default='gmm-test',
                      help='...(default: kmeans_result)')
  parser.add_argument('--epochs', dest='epochs', 
                      default=10, type=int,
                      help='the number of epochs(default: 10)')
  parser.add_argument('--n_cluster', dest='ncluster', required=False, 
                      nargs='+', type=int)

  args = parser.parse_args()


  s = get_files(args.indir)

  if "train"==args.mode:
    train(args.type, args.mdir, args.trained, s, args.epochs, args.ncluster)
  elif "predict"==args.mode:
    predict(args.mdir, args.outdir, args.trained, s, args.ncluster)
  elif "train_and_predict"==args.mode:
    train(args.type, args.mdir, args.trained, s, args.epochs, args.ncluster)
    predict(args.mdir, args.outdir, args.trained, s, args.ncluster)

  return 0


if __name__=='__main__':
  main()
