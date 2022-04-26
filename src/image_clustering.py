'''Usage
python src/image_clustering.py --mode train --type kmeans \
--input_dir '/data/kwkim/dataset/bladder/test_patches' \
--output_dir '/data/kwkim/aadw/results/test_20220425N0002/clustering' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans_test' \
--trained 'bladder-kmeans-test' --n_cluster 3 4 5 

python src/image_clustering.py --mode train --type kmeans \
--input_dir '/data/kwkim/aadw/dataset/bladder/patches-1024' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans/1024' \
--trained 'bladder-kmeans-1024' --n_cluster 16 32 64 --epochs 5

python src/image_clustering.py --mode train_and_predict --type kmeans \
--input_dir '/data/kwkim/aadw/dataset/bladder/patches-1024' \
--output_dir '/data/kwkim/aadw/results/patches-1024/clustering' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans/1024' \
--trained 'bladder-kmeans-1024' --n_cluster 16 32 64 --epochs 5

python src/image_clustering.py --mode predict --type kmeans \
--input_dir '/data/kwkim/aadw/dataset/bladder/test-1024' \
--output_dir '/data/kwkim/aadw/results/test-1024/clustering' \
--model_dir '/data/kwkim/aadw/pretrained/kmeans/1024' \
--trained 'bladder-kmeans-1024' --n_cluster 16 32 64 

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
import sys
sys.path.append("..")
from tqdm import tqdm
import numpy as np

import os
import cv2
import torch
import pickle
import argparse
from torch import optim, nn
from torchvision import models, transforms
from aadw.models.feature_extraction import FeatureExtractor


def get_model():
  # Initialize the model
  model = models.vgg16(pretrained=True)
  # model = models.resnet152(pretrained=True)
  new_model = FeatureExtractor(model)

  # Change the device to GPU
  device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
  new_model = new_model.to(device)
  return new_model, device


def get_transform():
  # Transform the image, so it becomes readable with the model
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()                
  ])
  return transform


def get_list(dir_: str) -> list:
    # list_ = []
    # for (dirpath, dirnames, filenames) in walk(dir_):
    #   list_.append(os.path.join(dirpath, f) for f in filenames)
    # return list_
    return [os.path.join(d, x) for d, _, files in os.walk(dir_) for x in files]


def extract_image_features(samples, new_model, transform, device):
  features = []
  for item in tqdm(samples, desc='Feature Extraction'):
    # Read the file
    img = cv2.imread(item)
    # Transform the image
    img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
    # Extract the feature from the image
      feature = new_model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
    features.append(feature.cpu().detach().numpy().reshape(-1))

  # Convert to NumPy Array
  features = np.array(features)
  return features

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


def dbscan(outdir:str, o_filename: str, samples, n_clusters, new_model, device, transform):
  print(f'[I] Input Dataset Size: {len(samples)}')
  features = extract_image_features(samples, new_model, transform, device)
  # DBSCAN
  from sklearn.cluster import DBSCAN
  # n_components로 미리 군집 개수 설정
  dbscan_kwargs = {
    'eps': 0.5,
    'min_samples': 12
  }
  print(f'[D] ========== GaussianMixture ==========')
  for n_cluster in n_clusters:
    dbscan = DBSCAN(**dbscan_kwargs)
    # gmm_labels = gmm.fit_predict(data)
    dbscan.fit(features)
    save_pkl(outdir, o_filename+'_'+str(n_cluster), dbscan)


def gmm(outdir:str, o_filename: str, samples, n_clusters, new_model, device, transform):
  print(f'[I] Input Dataset Size: {len(samples)}')
  features = extract_image_features(samples, new_model, transform, device)
  # GMM 적용
  from sklearn.mixture import GaussianMixture
  # n_components로 미리 군집 개수 설정
  gmm_kwargs = {
    # 'n_components': 3, 
    'random_state': 42,
  }
  print(f'[D] ========== GaussianMixture ==========')
  for n_cluster in n_clusters:
    gmm = GaussianMixture(n_components=n_cluster, **gmm_kwargs)
    # gmm_labels = gmm.fit_predict(data)
    gmm.fit(features)
    save_pkl(outdir, o_filename+'_'+str(n_cluster), gmm)


def kmeans(outdir:str, o_filename: str, samples, epochs, n_clusters, new_model, device, transform):
  ''' K-Means clustering
  '''
  print(f'[I] Input Dataset Size: {len(samples)}')
  kmeans_kwargs = {
    "init": "random",
    # "n_init": 10,
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
    features = extract_image_features(split, new_model, transform, device)
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

def train(type_: str, outdir:str, o_filename: str, samples, epochs, n_clusters, new_model, device, transform):
  if "kmeans"==type_:
    kmeans(outdir, o_filename, samples, epochs, n_clusters, new_model, device, transform)
  elif "gmm"==type_:
    gmm(outdir, o_filename, samples, n_clusters, new_model, device, transform)
  elif "dbscan"==type_:
    dbscan(outdir, o_filename, samples, n_clusters, new_model, device, transform)


def predict(samples, clustering_model, feature_extractor, device, transform):
  if 10000 < len(samples):
    n, _ = divmod(len(samples), 10000)
    split_list = np.array_split(samples, n)
  else:
    split_list = np.array_split(samples, 1)

  labels = []

  for split in tqdm(split_list, desc="Prediction(by split set)"):
    print(f'[D] subset type: {type(split)}, length: {len(split)}')
    features = extract_image_features(split, feature_extractor, transform, device)
    t_labels = clustering_model.predict(features)
    labels.extend(t_labels)

  return labels


def save(outdir, labels, samples):
  max_label = max(labels)
  # make output directory
  # if os.path.exists(outdir) is False:
  #   os.mkdir(outdir)
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
  parser.add_argument('--mode', dest='mode', choices=['train', 'predict', 'train_and_predict', 'debug', 'd2'],
                      default='train-and-predict', 
                      help='Operation mode [train|predict|train_and_predict] (default: predict)')
  parser.add_argument('--type', dest='type', choices=['kmeans', 'gmm', 'dbscan'],
                      default='kmeans', 
                      help='Clustering model type')
  parser.add_argument('--input_dir', dest='indir', 
                      # default='/data/kwkim/aadw/dataset/in',
                      default='/data/kwkim/aadw/dataset/bladder/test_patches',
                      help='Train Dataset Directory (default: /data/kwkim/aadw/dataset/in)')
  parser.add_argument('--output_dir', dest='outdir', 
                      # default='/data/kwkim/aadw/dataset/out',
                      default='/data/kwkim/aadw/results/test_clustering',
                      help='Result Directory (default: /data/kwkim/aadw/dataset/out)')
  parser.add_argument('--model_dir', dest='mdir', 
                      # default='/data/kwkim/aadw/result',
                      default='/data/kwkim/aadw/test_trained',
                      help='Result Directory (default: /data/kwkim/aadw/dataset/out)')
  parser.add_argument('--trained', dest='trained', 
                      # default='kmeans_result',
                      default='gmm-test',
                      help='...(default: kmeans_result)')
  parser.add_argument('--epochs', dest='epochs', 
                      default=10, type=int,
                      help='the number of epochs(default: 10)')
  parser.add_argument('--n_cluster', dest='ncluster', required=True, 
                      nargs='+', type=int)

  args = parser.parse_args()

  def _predict(args, s, n, d, t):
    for n_cluster in args.ncluster:
      model_filename = os.path.join(args.mdir, args.trained+'_'+str(n_cluster)+'.pkl')
      c_model = pickle.load(open(model_filename, "rb"))
      l = predict(s, c_model, n, d, t)
      save(args.outdir+'_'+str(n_cluster), l, s)

  n, d = get_model()
  t = get_transform()
  s = get_list(args.indir)

  if "train"==args.mode:
    train(args.type, args.mdir, args.trained, s, args.epochs, args.ncluster, n, d, t)
  elif "predict"==args.mode:
    _predict(args, s, n, d, t)
  elif "train_and_predict"==args.mode:
    train(args.type, args.mdir, args.trained, s, args.epochs, args.ncluster, n, d, t)
    _predict(args, s, n, d, t)

  return 0


if __name__=='__main__':
  main()
