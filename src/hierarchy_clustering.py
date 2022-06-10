'''
# Heirarchy clustering

## Method
### Heap Structure
Using Heap
Lv0/Lv1/  Lv2  /      Lv 3    
-----------------------------
|0|1, 2|3,4,5,6|7,8,9,10 ...
-----------------------------

### Node index
Parent node  = (child index) / 2
Left child node = (parent node)*2 + 1
Right child node = (parent node+1)*2

### Data in node
item = {"id": i, "image_dir": $(input_dir), "model": $(model file name)}

### Clustering
On working directory(current node), do clustering and save cluster's files in child node's image_dir
'''
'''
'''
import os
import pickle
import math
import json
import random
import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from utils.common import get_files
from utils.common import get_results
from models.feature_extraction import base_extractor
from models.feature_extraction import get_a_image_feature
from models.feature_extraction import save_npy
from models.feature_extraction import load_npy
from models.clustering import create_hierarchy


def arguments():
  parser = argparse.ArgumentParser(description='Image clustering')
  parser.add_argument('--mode', dest='mode', choices=['kmeans', 'gmm', 'birch'],
                      default='kmeans', 
                      help='Operation mode [kmeans|gmm|birch] (default: kmeans)')
  parser.add_argument('--input_dir', dest='indir', 
                      default='./input',
                      help='Train Dataset Directory (default: ./input)')
  parser.add_argument('--output_dir', dest='outdir', 
                      default='./output',
                      help='Result Directory (default: ./output)')
  parser.add_argument('--feature_name', dest='featurename', 
                      default='features',
                      help='Feature Name (default: features)')
  parser.add_argument('--model_dir', dest='modeldir', 
                      default='./output',
                      help='Model Directory (default: ./model)')
  parser.add_argument('--model_name', dest='modelname', 
                      default='model',
                      help='Model Name (default: model)')
  parser.add_argument('--level', type=int, default=10,
                      help='an integer for repatitive clustering depth level')
  args = parser.parse_args()
  return args


def get_features(img_info):
  features = dict()

  model, device, transform = base_extractor()
  # model, device, transform = base_extractor("vgg16")
  for item in tqdm(img_info, desc="Feature Extracting ..."):
    feature = get_a_image_feature(item["path"], model, device, transform)
    features[item["name"]] = feature

  return features


def save_pkl(outdir:str, o_filename:str, model):
  if not os.path.isdir(outdir):
    os.makedirs(outdir)
  o_filepath = os.path.join(outdir, f'{o_filename}.pkl')
  pickle.dump(model, open(o_filepath, "wb"), protocol=4)
 

def gmm(outdir:str, o_filename: str, total_features, images_info, epochs):
  from sklearn.mixture import GaussianMixture
  
  features = []
  for image in images_info:
    features.append(total_features[image["name"]])
  
  gmm_kwargs = {
    'n_components': 2, 
    'random_state': 42,
  }
  gmm = GaussianMixture(**gmm_kwargs)
  gmm.fit(features)          
  save_pkl(outdir, o_filename, gmm)


def birch(outdir:str, o_filename: str, total_features, images_info, epochs):
  # Extract Features
  features = []
  for image in images_info:
    features.append(total_features[image["name"]])

  # GMM 적용
  from sklearn.cluster import Birch
  # n_components로 미리 군집 개수 설정
  birch_kwargs = {
    'n_clusters': 2, 
  }
  birch = Birch(**birch_kwargs)
  birch.fit(features)
  save_pkl(outdir, o_filename, birch)


def gmm(outdir:str, o_filename: str, total_features, images_info, epochs):
  # Extract Features
  features = []
  for image in images_info:
    features.append(total_features[image["name"]])

  # GMM 적용
  from sklearn.mixture import GaussianMixture
  # n_components로 미리 군집 개수 설정
  gmm_kwargs = {
    'n_components': 2, 
    'random_state': 42,
  }
  gmm = GaussianMixture(**gmm_kwargs)
  gmm.fit(features)
  save_pkl(outdir, o_filename, gmm)


def kmeans(outdir:str, o_filename: str, total_features, images_info, epochs):
  ''' K-Means clustering
  '''
  print(f'[I] Input Dataset Size: {len(images_info)}')
  kmeans_kwargs = {
    "init": "random",
    "max_iter": 300,
    "random_state": 42,
  }

  if 10000 < len(images_info):
    n, _ = divmod(len(images_info), 10000)
    split_list = np.array_split(images_info, n)
  else:
    split_list = np.array_split(images_info, 1)

  # Extract Features
  features_list = []
  # for split in tqdm(split_list, desc="Feature Extracting..."):
  for split in split_list:
    features = []
    for image in split:
      features.append(total_features[image["name"]])
    features_list.append(features)


  # Get Model and learning
  if len(split_list) == 1:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
    epochs_ = 1
  else:
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=2, **kmeans_kwargs)
    epochs_ = epochs
    
  for _ in tqdm(range(epochs_), desc='K-Means Learning'):
    for features in features_list:
      if len(split_list) == 1:
        kmeans.fit(features)          
      else:
        kmeans.partial_fit(features) ## Partially fitting data in batches

    save_pkl(outdir, o_filename, kmeans)


def is_directory(dir_):
  if not os.path.isdir(dir_):
    os.makedirs(dir_)


def save(saveflag, labels, images_info, outdir, left, right):
  is_directory(left)
  is_directory(right)

  if saveflag:
    from shutil import copyfile
    for idx, filename in enumerate(tqdm(labels, desc='Copy file into cluster')):
      label = labels[filename]
      if label == 0:
        copyfile(images_info[idx]["path"], os.path.join(outdir, left, filename))
      else:
        copyfile(images_info[idx]["path"], os.path.join(outdir, right, filename))

  # save a image list file
  images_left = list()
  images_right = list()
  for idx, filename in enumerate(tqdm(labels, desc='Copy file into cluster')):
    label = labels[filename]
    element = {"name": images_info[idx]["name"], 
              "path": images_info[idx]["path"]}
    if label == 0:
        images_left.append(element)
    else:
      images_right.append(element)

  with open(os.path.join(outdir, left, "image_list.txt"), "w") as f:
    j = json.dumps({"images": images_left})
    f.write(j)
  with open(os.path.join(outdir, right, "image_list.txt"), "w") as f:
    j = json.dumps({"images": images_right})
    f.write(j)


def _predict(total_features, images_info, clustering_model):
  labels = dict()

  features = []
  for image in images_info:
    features.append(total_features[image["name"]])
  logger.info('working for prediction ...')
  start = time.time()
  t_labels = clustering_model.predict(features)
  end = time.time()
  logger.debug(f'Elapsed time: {end - start} / working with {len(features)} images')

  for idx, image in enumerate(images_info):
    labels[image["name"]] = int(t_labels[idx])

  return labels


def predict(saveflag:bool, model_dir, outdir, o_filename, total_feature, images_info, left, right):
  model_filename = os.path.join(model_dir, o_filename+'.pkl')
  c_model = pickle.load(open(model_filename, "rb"))
  labels = _predict(total_feature, images_info, c_model)
  save(saveflag, labels, images_info, outdir, left, right)
  return labels


METRICS = {
  "silhouette": silhouette_score,
  "dbindex": davies_bouldin_score,
  "chindex": calinski_harabasz_score
}
def cluster_metric(total_features, images_info, label_info, input_dir, 
                   saveflag:bool=True, type_:str="silhouette"):
  # get features and labels
  features = []
  labels = []
  for image in images_info:
    features.append(total_features[image["name"]])
    labels.append(label_info[image["name"]])
  
  start = time.time()
  score = METRICS[type_](features, labels)
  end = time.time()
  elapsed = end - start

  logger.debug(f"Elapsed time(Metric): {elapsed}")

  if saveflag:
    save_results(label_info, type_, score, input_dir)

  return score


def save_results(label_info, type_, score, input_dir):
  import json
  result = {"results": label_info, type_: score}
  with open(os.path.join(input_dir, "clustering-results.txt"), "w") as f:
    f.write(json.dumps(result))


def save_thumbnail(image_info, labels, input_dir, nodename):
  left, right = [], []
  for name, label in labels.items():
    if label == 0:
      left.append(name)
    elif label == 1:
      right.append(name)
    else:
      logger.warning(f"Error label\tname: {name}\tlabel: {label}")

  # image list to dict
  image_dict = dict()
  for i in image_info:
    image_dict[i["name"]] = i["path"]

  col_no = 10 # the number of column
  raw_no = 10 # the number of raw
  width = 102 # the wight of each image 
  height = 122 # the height of each image
  concatenated_image = Image.new("RGB", (col_no*width, raw_no*height), "white")
  
  for i in range(2):
    if 0==i:
      # 1. upper side(left)
      target = left
    else:
      # 2. below side(right)
      target = right 

    sample_idx = list()
    if 50>len(target):
      while (50>=len(sample_idx)):
        sample_idx.extend([x for x in range(len(target))])
      sample_idx = sample_idx[:50]
    else:
      sample_idx = random.sample(range(len(target)), 50)
    
    for idx, image_idx in enumerate(sample_idx):
      img = Image.open(image_dict[target[image_idx]])
      img = img.resize((width, height))
      concatenated_image.paste(img, (((idx+(i*50))%col_no)*width, 
                                     ((idx+(i*50))//raw_no)*height))

  # add text
  if nodename == "root":
    cur_node = ''
  else:
    cur_node = nodename
  font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
  font = ImageFont.truetype(font_path, size=48)
  ImageDraw.Draw(
      concatenated_image  # Image
  ).text(
    (0, 0),  # Coordinates
    cur_node+'A',  # Text
    (57, 255, 20),  # Color(Neon)
    font=font
  )
  ImageDraw.Draw(
      concatenated_image  # Image
  ).text(
    (0, height*5),  # Coordinates
    cur_node+'B',  # Text
    (57, 255, 20),  # Color(Neon)
    font=font
  )
  # save final thumbnail image
  concatenated_image.save(os.path.join(input_dir, f'thumbnail_{nodename}.png'))


CLUSTER_MODEL = {
  "kmeans": kmeans,
  "gmm": gmm,
  "birch": birch
}
def run_main(mode, total_features, element:dict, element_next:dict, 
             element_after_next:dict, saveflag):
  input_dir = element["image_dir"]
  pretraied_name = element["model"]
  images_info = get_files(input_dir, type="both")
  if 0==len(images_info):
    images_info = get_results(input_dir)
  
  logger.debug(f"working images: {len(images_info)}")

  if len(images_info)<50: return False
  # Clustering
  CLUSTER_MODEL[mode](model_dir, pretraied_name, total_features, images_info, 20)

  left_image_dir = element_next["image_dir"]
  right_image_dir = element_after_next["image_dir"]

  # Prediction
  labels = predict(saveflag, model_dir, left_image_dir, pretraied_name, 
      total_features, images_info, left_image_dir, right_image_dir)
  # Metric
  metric_type = 'chindex'
  score = cluster_metric(total_features, images_info, labels, input_dir, type_=metric_type)
  logger.debug(f"Cluster metric({metric_type}) score: {score}")
  # save thumbnail
  save_thumbnail(images_info, labels, input_dir, element["name"])

  return True

''' notes
[Repatitive Clustering]

result dir: /data/kwkim/aadw/results/bladder_kmeans_v3.0
<result>
(20220607N002): kmeans with stain normalization(HE)
(20220607N003): kmeans without stain normalization
(20220610N001): kmeans. applied nasnetlarge architecture
'''
''' USAGE
python src/hierarchy_clustering.py --mode kmeans \
--input_dir /data/kwkim/dataset/bladder/trainset_v3.0 \
--output_dir /data/kwkim/aadw/results/bladder_kmeans_v3.0/20220610N001/cluster \
--feature_name features
--model_dir /data/kwkim/aadw/pretrained/kmeans/bladder_kmeans_v3.0_20220610N001 \
--model_name bladder_kmeans_v3.0_20220610N001 \
--level 10

tree /data/kwkim/aadw/results/bladder_kmeans_v3.0/20220607N001
tree /data/kwkim/aadw/pretrained/kmeans/bladder_kmeans_v3.0_20220607N001

rm -r /data/kwkim/aadw/results/bladder_kmeans_v3.0/20220607N001
rm -r /data/kwkim/aadw/pretrained/kmeans/bladder_kmeans_v3.0_20220607N001
'''
''' TEST
python src/hierarchy_clustering.py --mode birch \
--input_dir /data/kwkim/dataset/bladder/test_patches \
--output_dir /data/kwkim/aadw/results/h_test/cluster \
--model_dir /data/kwkim/aadw/pretrained/h_test/20220607N001 \
--model_name birch \
--level 4

tree /data/kwkim/aadw/results/h_test
tree /data/kwkim/aadw/pretrained/h_test/20220607N001

rm -r /data/kwkim/aadw/results/h_test
rm -r /data/kwkim/aadw/pretrained/h_test/20220607N001
'''
from utils.log import make_logger
if __name__=="__main__":
  args = arguments()
  global logger
  logger = make_logger("hierarchy_cluster", level="info")
  logger.info("START HIERARCHY CLUSTER")

  mode = args.mode
  input_dir = args.indir
  output_dir = args.outdir
  model_dir = args.modeldir
  model_name = args.modelname
  level = args.level
  
  heap = create_hierarchy(input_dir, model_name, output_dir, level)
  logger.info("get hierarchy structure")

  ''' heap
    왼쪽 자식의 인덱스 = (부모의 인덱스) * 2 +1
    오른쪽 자식의 인덱스 = (부모의 인덱스 + 1) * 2 
    부모의 인덱스 = (자식의 인덱스-1) // 2
  '''
  total_features = dict()
  # extract features from original image directory
  input_dir = heap[0]["image_dir"]
  images_info = get_files(input_dir, type="both")
  # total_features = load_npy(os.path.join(input_dir, "HE-features.npy"))
  total_features = load_npy(os.path.join(input_dir, args.featurename))
  if total_features is None:
    total_features = get_features(images_info)
    save_npy(total_features, input_dir, name=args.featurename)
  logger.info("extract features")
  logger.debug(f"features: {len(total_features)}")
  
  # root
  saveflag = False
  res = run_main(mode, total_features, heap[0], heap[1], heap[2], saveflag)
  if res is False: exit(0)
  # child
  for l in tqdm(range(level-1), desc="Clustering ..."):
    for i in range(int(math.pow(2, l+1))-1, int(math.pow(2, l+2)-1)):
      logger.debug(f"level: {l}\tworking node: {i}")
      saveflag = True if l >= (level-3) else False
      res = run_main(mode, total_features, heap[i], heap[i*2+1], heap[(i+1)*2], saveflag)
      if res is False: continue
      