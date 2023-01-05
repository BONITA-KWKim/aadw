import os
import sys
import math
import numpy as np
import random
import torch
import torchvision

from time import time
from absl import (app, flags, logging)
from openslide import OpenSlide
from PIL import Image, ImageDraw
from itertools import permutations
from tqdm import tqdm

from utils.visualize_map import WSIMap
from models.bladder_classification import BCNet

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')


def init_ai():
  logging.debug('init_ai()')
  PATH = '/data/kwkim/aadw/models/bladder_classification/BCNet_best.pth'

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = BCNet.BladderClassifier(4, 512)
  model.load_state_dict(torch.load(PATH))

  model.to(device)

  return device, model


def get_transform(transform=None):
  if transform:
    transform = transform
  else:
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
  return transform


def prepare_each_image(data, transform, slice_type:str='vertical', 
                       slice_count:int=4, make_permutations:bool=False):
  assert transform is not None, 'Transform is None'
  #data = cv.cvtColor(data, cv.COLOR_BGR2HSV)
  size = data.shape[0] // slice_count
  assert data.shape[0] == data.shape[1], \
    'Image must have same width and height.: ' + str(data.shape)
  cropped_data_list = []

  if slice_count == 0:
      data = transform(data)
      return data

  if slice_type == 'both':
    for i in range(slice_count):
      row_data = []
      for j in range(slice_count):
        cropped_data = data[i*size:i *
                            size+size, j*size:j*size+size]
        cropped_data = transform(cropped_data)
        row_data.append(cropped_data)
      cropped_data_list.append(row_data)

  elif slice_type == 'vertical':
    for i in range(slice_count):
      cropped_data = data[0:data.shape[0], i*size:i*size+size]
      cropped_data = transform(cropped_data)
      cropped_data_list.append(cropped_data)

  elif slice_type == 'horizontal':
    for i in range(slice_count):
      cropped_data = data[i*size:i*size+size, 0:data.shape[1]]
      cropped_data = transform(cropped_data)
      cropped_data_list.append(cropped_data)
  else:
    logging.error('Wrong slice type!!')
    assert False

  if not make_permutations:
    if slice_type == 'both':
      temp = []
      for i in range(slice_count):
        temp += cropped_data_list[i]
      temp = torch.stack(temp)
      return temp
    else:
      return torch.stack(cropped_data_list)
  else:
    if slice_type == 'both':
      new_list = []
      for i in range(len(cropped_data_list)):
        cropped_data_list[i] = list(permutations(
          cropped_data_list[i], slice_count))
      for i in range(len(cropped_data_list[0])):
        data = []
        for j in range(slice_count):
          data += list(cropped_data_list[j][i])
        new_list.append(torch.stack(data))
      cropped_data_list = new_list
    else:
      cropped_data_list = list(permutations(cropped_data_list, slice_count))
    
    for data in cropped_data_list:
      if slice_type == 'both':
        tensor_data = data
      else:
        tensor_data = torch.stack(data)
      return tensor_data


def random_class():
  # sample_class = [0, 1, 2, 3, 4]
  sample_class = [0, 1, 2]
  return random.choice(sample_class)


def inference(device, model, img):
  img = img.to(device)
  output = model(img)
  pred_labels = output[0]
  p = pred_labels.item()

  return p


def get_colour(no):

  def _color_to_np_color(color: str, transparent:int=100) -> np.ndarray:
    """
    Convert strings to NumPy colors.
    Args:
        color: The desired color as a string.
    Returns:
        The NumPy ndarray representation of the color.
    """
    colors = {
      "white": np.array([255, 255, 255, transparent]),
      "pink": np.array([255, 108, 180, transparent]),
      "black": np.array([0, 0, 0, transparent]),
      "red": np.array([255, 0, 0, transparent]),
      "purple": np.array([225, 225, 0, transparent]),
      "yellow": np.array([255, 255, 0, transparent]),
      "orange": np.array([255, 127, 80, transparent]),
      "blue": np.array([0, 0, 255, transparent]),
      "green": np.array([0, 255, 0, transparent])  }
    return colors[color]

  c = {
    0: _color_to_np_color('black'),
    1: _color_to_np_color('green'),
    2: _color_to_np_color('purple'),
    3: _color_to_np_color('blue'),
    4: _color_to_np_color('orange'),
  }
  return c[no]


def get_offset(openslide, window_size:int=32) -> tuple:
  # logging.debug(f'WSI properties\n{openslide.properties}')
  
  width = openslide.level_dimensions[0][0]
  height = openslide.level_dimensions[0][1]
  logging.debug(f'Level dimension: {openslide.level_dimensions[0]}')

  w_count = math.floor(width/window_size)
  h_count = math.floor(height/window_size)
  logging.debug(f'Count. W: {w_count}, Y: {h_count}')

  w_offset = width-(w_count*window_size)
  h_offset = height-(h_count*window_size)
  offset = (math.floor(w_offset/2), math.floor(h_offset/2)) # (w,h)
  logging.debug(f'Offset: {offset}')

  xx = []
  for i in range(w_count):
    xx.append(window_size*i+offset[0])
  logging.debug(f'xx: {len(xx)}. Start: {xx[0]}, Stop: {xx[-1]}')

  yy = []
  for i in range(h_count):
    yy.append(window_size*i+offset[1])
  logging.debug(f'yy: {len(yy)}. Start: {yy[0]}, Stop: {yy[-1]}')

  return width, height, xx, yy, w_count, h_count


def get_image_mean_and_std(img) -> tuple:
  N_CHANNELS = 3
  mean = [.0, .0, .0]
  std = [.0, .0, .0]
  for i in range(N_CHANNELS):
    if isinstance(img, np.ndarray):
      I = img
    else:
      I = np.asarray(img.convert('RGB'))
    mean[i] = round(I[:,:,i].mean(), 4)
    std[i] = round(I[:,:,i].std(), 4)
  
  # logging.debug(f'Mean: {np.array(mean).mean()}, Std: {np.array(std).mean()}')
  return np.array(mean).mean(), np.array(std).mean()


log_level = {
  'fatal': logging.FATAL,
  'error': logging.ERROR,
  'warning': logging.WARNING,
  'info': logging.INFO,
  'debug': logging.DEBUG
}
def main(argv):
  # %% Init
  del argv
  logging.set_verbosity(log_level['debug'])

  ROOT = os.path.dirname(os.path.abspath(__file__))
  OPENSLIDE_PATH = 'test_imgs/breast-test.svs'
  SEG_LEVEL = 0
  WINDOW_SIZE = 32
  SHRINK_RATIO = 10
  #MAG_SIZE = 7 # window_size(32) * mag_size(7) = real tile size(224)
  MAG_SIZE = 16 # window_size(32) * mag_size(16) = real tile size(512)
  
  logging.info(f'Initialization\r\n\tTarget Image: {OPENSLIDE_PATH}\
    \r\n\tImage information. size: {(WINDOW_SIZE*MAG_SIZE)}, window: {WINDOW_SIZE}, window step: {MAG_SIZE}\
    \r\n\tTissue Detection Downsample ratio: {SHRINK_RATIO}')

  # %% open WSI
  ''' Open WSI 
  '''
  t0 = time()
  wsimap = WSIMap(log_level['debug'], os.path.join(ROOT, OPENSLIDE_PATH), 
                  speciman_type='TEST', window_size=WINDOW_SIZE)
  
  width, height = wsimap.get_size()
  xx, yy = wsimap.get_map_coordinate()
  w_count, h_count = wsimap.get_map_count()
  t1 = time()
  logging.info(f'Open WSI (elapsed: {round(t1-t0, 4)}s).')

  # %% Tissue Detection
  ''' Tissue Detection
  '''
  t0 = time()
  t_width = width//SHRINK_RATIO
  t_height = height//SHRINK_RATIO
  thumbnail_image = wsimap.get_thumbnail(t_width, t_height) # PIL image
  t1 = time()
  logging.info(f'Get thumbnail image (elapsed: {round(t1-t0, 4)}s). Type: {type(thumbnail_image)}')

  t0 = time()
  t_np = np.array(thumbnail_image.convert('RGB'))
  for i in range(0, w_count, MAG_SIZE):
    for j in range(0, h_count, MAG_SIZE):
      # numpy shape = (h, w, c)
      img_tile = t_np[yy[j]//SHRINK_RATIO:(yy[j]+(wsimap.window_size*MAG_SIZE))//SHRINK_RATIO,
                      xx[i]//SHRINK_RATIO:(xx[i]+(wsimap.window_size*MAG_SIZE))//SHRINK_RATIO, :]
      m, s = get_image_mean_and_std(img_tile)
      # tissue detection
      if m < 230 and s > 5.0:
        for ii in range(MAG_SIZE):
           if (i+ii) > (w_count-1): continue
           for jj in range(MAG_SIZE):
            if (j+jj) > (h_count-1): continue
            wsimap.info_map[i+ii][j+jj].active = True
  t1 = time()
  logging.info(f'Detect active area (elapsed: {round(t1-t0, 4)}s). window size: {wsimap.window_size}, magnification: {MAG_SIZE}')
  
  # %% AI analysis
  ''' AI analysis
  '''
  t0 = time()
  seglevel_image = wsimap.get_total_image_by_level(SEG_LEVEL) # PIL image
  seglevel_image = seglevel_image.convert('RGB') # PIL image is RGBA. Need to convert
  t1 = time()
  logging.info(f'Read total image (elapsed: {round(t1-t0, 4)}s). \
    Type: {type(seglevel_image)}. Memory size: {sys.getsizeof(seglevel_image)}')
  logging.debug(f'total image({seglevel_image.size}). \
    w: {seglevel_image.width}. h: {seglevel_image.height}.')

  t0 = time()
  seglevel_image_np = np.array(seglevel_image)
  t1 = time()
  logging.info(f'Level {SEG_LEVEL} total image (elapsed: {round(t1-t0, 4)}s). \
    Type: {type(seglevel_image_np)}. \
    Memory size: {sys.getsizeof(seglevel_image_np)}')

  device, model = init_ai()

  t0 = time()
  # for i in range(0, w_count, MAG_SIZE):
  #   for j in range(0, h_count, MAG_SIZE):
  for i in tqdm(range(0, w_count, MAG_SIZE//4)):
    for j in tqdm(range(0, h_count, MAG_SIZE//4), leave=False):
      # get region
      # loc = (xx[i], yy[j])
      # region_size = (xx[i]+(wsimap.window_size*MAG_SIZE), yy[j]+(wsimap.window_size*MAG_SIZE))
      # r = wsimap.get_tile(loc, SEG_LEVEL, region_size)
      region = seglevel_image_np[yy[j]:yy[j]+(wsimap.window_size*MAG_SIZE),  
                                  xx[i]:xx[i]+(wsimap.window_size*MAG_SIZE), :]
      # (warning) 마지막에 (225, 512, 3)으로 읽힘
      # logging.debug(f'Region. type: {type(region)}. shape: {region.shape}')

      # inference
      if region.shape[0] != region.shape[1]: 
        logging.warning(f'Region. type: {type(region)}. shape: {region.shape}')
        continue
      #class_ = random_class() # TEST
      t = get_transform()
      img = prepare_each_image(region, t) 
      img = img.unsqueeze(0)
      class_ = inference(device, model, img)

      if 1 == class_: continue # do not add pseudo label normal(1)

      # save results information
      for ii in range(MAG_SIZE):
        if (i+ii) > (w_count-1): continue
        for jj in range(MAG_SIZE):
          if (j+jj) > (h_count-1): continue
          wsimap.info_map[i+ii][j+jj].classification.append(class_)
  t1 = time()
  logging.info(f'AI analysis (elapsed: {round(t1-t0, 4)}s)')
  
  # %% Visualization
  ''' Visualization
  '''
  t0 = time()
  voted_count_list = []
  draw = ImageDraw.Draw(thumbnail_image, 'RGBA')
  for i in tqdm(range(w_count), desc='Visualization ...'):
    for j in tqdm(range(h_count), leave=False):
      voted, voted_cnt = wsimap.info_map[i][j].get_voted()
      if voted_cnt not in voted_count_list: voted_count_list.append(voted_cnt) # debug
      # if 0 == voted or None == voted or 15 > voted_cnt: continue
      if None == voted: continue
      colour = get_colour(voted)
      if wsimap.info_map[i][j].active == True:
        draw.ellipse((xx[i]//SHRINK_RATIO, yy[j]//SHRINK_RATIO, 
                     (xx[i]+wsimap.window_size)//SHRINK_RATIO, 
                     (yy[j]+wsimap.window_size)//SHRINK_RATIO),
                     fill = (colour[0], colour[1], colour[2], colour[3]))
  logging.debug(f'voted count: {voted_count_list}') # debug
  thumbnail_image.save(f'results/visualize_test-x{MAG_SIZE}-w{wsimap.window_size}.png')
  t1 = time()
  logging.info(f'Visualization (elapsed: {round(t1-t0, 4)}s). visualize_test-x{MAG_SIZE}-w{wsimap.window_size}')


# %% main
if __name__=="__main__":
  app.run(main)
