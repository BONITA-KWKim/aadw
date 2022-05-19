import os
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils.common import get_files
from utils.common import get_dirs
from utils.common import create_output_directory
from models.feature_extraction import get_image_features


# t-SNE 시각화 함수 정의
def plot_vecs_n_labels(v, fname):
  plt.figure(figsize = (10,10))
  plt.axis('off')
  sns.set_style('darkgrid')
  sns.scatterplot(v[:,0], v[:,1], legend='full', palette=sns.color_palette("bright", 10))
  plt.savefig(fname)


def make_tnse(dir_:str):
  samples = get_files(dir_)
  features = get_image_features(samples)

  # 2차원으로 차원 축소
  n_components = 2
  tsne = TSNE(n_components=n_components)
  # 모델의 출력값을 tsne.fit_transform에 입력하기
  pred_tsne = tsne.fit_transform(features)

  # t-SNE 시각화 함수 실행
  plot_vecs_n_labels(pred_tsne, 'tsen_bladder.png')


'''
params:
1) display_no
2) display_size
3) input_dir
4) output_dir
'''
def get_tile_info(total_tile_no, display_size):
  # with: x, height 1.6x => get raw/col count
  col_count = int(math.floor(math.sqrt(total_tile_no/1.6)))
  raw_count = int(math.floor(total_tile_no/col_count))
  if col_count*raw_count < total_tile_no:
    raw_count += 1
  width = int(display_size[0]/col_count)
  height = int(display_size[1]/raw_count)

  return col_count, raw_count, width, height


def display_clustering(indir: str, outdir:str, display_no: int, display_size):
  if isinstance(display_size, int):
    o_size = (display_size, int(display_size*1.6))
  elif isinstance(display_size, tuple):
    o_size = display_size
  else:
    print(f'[E] {type(display_size)} is invalid type')
  
  sub_dirs = get_dirs(indir)

  ''' file map description
  file_map = {
    "sub_dir_name": [$(image_names)],
    ...
  }
  '''
  file_map = dict()
  for s in sub_dirs:
    # imgs = get_files(os.path.join(indir, s), type="name")
    imgs = get_files(os.path.join(indir, s))
    file_map[s] = imgs

  create_output_directory(outdir)

  for i in tqdm(range(display_no)):
    cluster_no = len(sub_dirs)

    col_count, raw_count, width, height = get_tile_info(cluster_no, o_size)
    concatenated_image = Image.new("RGB", (width*col_count, height*raw_count), "white")

    for idx in range(col_count*raw_count):
      if idx < cluster_no:
        cluster_name = sub_dirs[idx]
        # pick random image file
        pick = random.randint(0, len(file_map[cluster_name])-1)
        # open image file and resize
        img = Image.open(file_map[cluster_name][pick])
        img = img.resize((width, height))
      else:
        # create dummy image
        img = Image.new("RGB", (width, height), "black")
      # concatenate
      concatenated_image.paste(img, ((idx%col_count*width), (idx//col_count*height)))

    # save image file
    concatenated_image.save(os.path.join(outdir, f'{i}.png'))

  return 0


if __name__=="__main__":
  # indir = '/data/kwkim/aadw/results/bladder_kmeans_v2.1/20220518N001/cluster_80'
  # outdir = '/data/kwkim/visual_results/bladder_kmeans_v2.1/20220518N001/cluster_80'
  indir = '/data/kwkim/aadw/results/bladder_kmeans_v1.0/20220518N001/cluster_48'
  outdir = '/data/kwkim/visual_results/bladder_kmeans_v1.0/20220518N001/cluster_48'
  indir1 = '/data/kwkim/aadw/results/bladder_kmeans_v1.0/20220518N001/cluster_80'
  outdir1 = '/data/kwkim/visual_results/bladder_kmeans_v1.0/20220518N001/cluster_80'
  indir2 = '/data/kwkim/aadw/results/bladder_kmeans_v1.0/20220518N001/cluster_99'
  outdir2 = '/data/kwkim/visual_results/bladder_kmeans_v1.0/20220518N001/cluster_99'
  i_list = [indir, indir1, indir2]
  o_list = [outdir, outdir1, outdir2]
  display_no = 20
  display_size = (800, 1280)
  # display_clustering(indir, outdir, display_no, display_size)
  for i in range(len(i_list)):
    print('[I] Current working directory')
    print(f'\tinput: {i_list[i]}')
    print(f'\toutput: {o_list[i]}')
    display_clustering(i_list[i], o_list[i], display_no, display_size)
    

  # dir_ = ''
  # make_tnse(dir_)