import math
import numpy as np
import random

from time import time
from absl import app
from absl import flags
from absl import logging
from openslide import OpenSlide
from PIL import Image, ImageDraw

from utils.visualize_map import WSIMap

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')


def random_class():
  # sample_class = [0, 1, 2, 3, 4]
  sample_class = [0, 1, 2]
  return random.choice(sample_class)


def get_colour(no):
  c = {
    0: color_to_np_color('black'),
    1: color_to_np_color('green'),
    2: color_to_np_color('purple'),
    3: color_to_np_color('blue'),
    4: color_to_np_color('orange'),
  }
  return c[no]

def color_to_np_color(color: str, transparent:int=100) -> np.ndarray:
  """
  Convert strings to NumPy colors.
  Args:
      color: The desired color as a string.
  Returns:
      The NumPy ndarray representation of the color.
  """
  colors = {
    # "white": np.array([255, 255, 255]),
    # "pink": np.array([255, 108, 180]),
    # "black": np.array([0, 0, 0]),
    # "red": np.array([255, 0, 0]),
    # "purple": np.array([225, 225, 0]),
    # "yellow": np.array([255, 255, 0]),
    # "orange": np.array([255, 127, 80]),
    # "blue": np.array([0, 0, 255]),
    # "green": np.array([0, 255, 0])
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
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
# OPENSLIDE_PATH = 'test_imgs/test.svs'
OPENSLIDE_PATH = 'test_imgs/test001.svs'
def main(argv):
  del argv

  logging.set_verbosity(log_level['debug'])

  SEG_LEVEL = 0
  WINDOW_SIZE = 32
  SHRINK_RATIO = 10
  MAG_SIZE = 7 # window_size(32) * mag_size(7) = real tile size(224)
  ''' Open WSI 
  '''
  t0 = time()
  wsimap = WSIMap(log_level['debug'], os.path.join(ROOT, OPENSLIDE_PATH), 
                  speciman_type='TEST', window_size=WINDOW_SIZE)
  
  width, height = wsimap.get_size()
  xx, yy = wsimap.get_map_coordinate()
  w_count, h_count = wsimap.get_map_count()
  t1 = time()
  logging.debug(f'Open WSI (elapsed: {round(t1-t0, 4)}s).')

  ''' Tissue Detection
  '''
  t0 = time()
  t_width = width//SHRINK_RATIO
  t_height = height//SHRINK_RATIO
  thumbnail_image = wsimap.get_thumbnail(t_width, t_height) # PIL image
  t1 = time()
  logging.debug(f'Get thumbnail image (elapsed: {round(t1-t0, 4)}s). Type: {type(thumbnail_image)}')

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
  logging.debug(f'Detect active area (elapsed: {round(t1-t0, 4)}s). window size: {wsimap.window_size}, magnification: {MAG_SIZE}')
  
  ''' AI analysis
  '''
  t0 = time()
  seglevel0_image = wsimap.get_total_image_by_level(SEG_LEVEL)
  t1 = time()
  logging.debug(f'Read total image (elapsed: {round(t1-t0, 4)}s). Type: {type(seglevel0_image)}. Memory size: {sys.getsizeof(seglevel0_image)}')

  t0 = time()
  seglevel0_image_np = np.array(seglevel0_image)
  t1 = time()
  logging.debug(f'Level {SEG_LEVEL} total image (elapsed: {round(t1-t0, 4)}s). Type: {type(seglevel0_image_np)}. Memory size: {sys.getsizeof(seglevel0_image_np)}')

  t0 = time()
  for i in range(0, w_count, MAG_SIZE):
    for j in range(0, h_count, MAG_SIZE):
      # get region
      # loc = (xx[i], yy[j])
      # region_size = (xx[i]+(wsimap.window_size*MAG_SIZE), yy[j]+(wsimap.window_size*MAG_SIZE))
      # r = wsimap.get_tile(loc, SEG_LEVEL, region_size)
      region = seglevel0_image_np[yy[j]:yy[j]+(wsimap.window_size*MAG_SIZE),  
                                  xx[i]:xx[i]+(wsimap.window_size*MAG_SIZE), :]
      # inference
      class_ = random_class()
      # save results information
      for ii in range(MAG_SIZE):
        if (i+ii) > (w_count-1): continue
        for jj in range(MAG_SIZE):
          if (j+jj) > (h_count-1): continue
          wsimap.info_map[i+ii][j+jj].classification.append(class_)
  t1 = time()
  logging.debug(f'AI analysis (elapsed: {round(t1-t0, 4)}s)')
  
  ''' Visualization
  '''
  t0 = time()
  # colour = color_to_np_color('green', transparent=80)
  draw = ImageDraw.Draw(thumbnail_image, 'RGBA')
  for i in range(w_count):
    for j in range(h_count):
      voted, _ = wsimap.info_map[i][j].get_voted()
      colour = get_colour(voted)
      if wsimap.info_map[i][j].active == True:
        draw.ellipse((xx[i]//SHRINK_RATIO, yy[j]//SHRINK_RATIO, 
                     (xx[i]+wsimap.window_size)//SHRINK_RATIO, 
                     (yy[j]+wsimap.window_size)//SHRINK_RATIO),
                     fill = (colour[0], colour[1], colour[2], colour[3]))
  thumbnail_image.save(f'results/visualize_test-x{MAG_SIZE}-w{wsimap.window_size}.png')
  t1 = time()
  logging.debug(f'Visualization (elapsed: {round(t1-t0, 4)}s). visualize_test-x{MAG_SIZE}-w{wsimap.window_size}')


if __name__=="__main__":
  app.run(main)
