import os
import math
import numpy as np
from openslide import OpenSlide
from absl import logging

class WSIElement:

  def __init__(self, x, y):
    self.test = 'test'
    self.test_x = x
    self.test_y = y

    self.active = False
    self.classification = list()
    self.voted = None
    self.voted_cnt = 0
  
  def get_test(self):
    return f'{self.test}_({self.test_x}, {self.test_y})'


  def get_voted(self):
    self.voting()
    return self.voted, self.voted_cnt

  
  def voting(self):
    vals, counts = np.unique(self.classification, return_counts=True)
    index = np.argmax(counts)
    self.voted = vals[index]
    self.voted_cnt = counts[index]


class WSIMap:
  
  def __init__(self, log_lv, path, speciman_type:str='cervix', window_size:int=32):
    logging.set_verbosity(log_lv)
    self.openslide = None
    self.window_size = window_size
    self.width = 0
    self.height = 0
    self.xx = []
    self.yy = []
    self.w_count = 0
    self.h_count = 0

    self.speciman_type = speciman_type
    self.info_map = None

    try:
      self.openslide = OpenSlide(path)
    except:
      print(f'Could not open: {path}')
    
    self.flag_file_opened = True if self.openslide is not None else False
    self.get_offset(self.window_size)
    self.init_info_map(self.xx, self.yy)


  def is_wsi_opened(self):
    return self.flag_file_opened 


  def get_offset(self, window_size):
    if self.flag_file_opened == False: return None
    # logging.debug(f'WSI properties\n{openslide.properties}')
    
    width = self.openslide.level_dimensions[0][0]
    height = self.openslide.level_dimensions[0][1]
    logging.debug(f'Level dimension: {self.openslide.level_dimensions[0]}')

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

    # save results
    self.width = width
    self.height = height
    self.xx = xx
    self.yy = yy
    self.w_count = w_count
    self.h_count = h_count
  

  def init_info_map(self, xx, yy):
    x_len = 0
    y_len = 0
    if isinstance(xx, list):
      x_len = len(xx)
    if isinstance(yy, list):
      y_len = len(yy)

    self.info_map = np.empty(shape=(x_len, y_len), dtype=object)
    for x in range(x_len):
      for y in range(y_len):
        el = WSIElement(x, y)
        self.info_map[x][y] = el

    logging.debug(f'Information Map. shape: {self.info_map.shape}')


  def get_size(self):
    return self.width, self.height


  def get_map_count(self):
    return self.w_count, self.h_count


  def get_map_coordinate(self):
    return self.xx, self.yy


  def get_thumbnail(self, t_width:int, t_height:int):
    return self.openslide.get_thumbnail((t_width, t_height))
    

  def get_tile(self, loc:tuple, level: int, size:tuple):
    if len(loc) != 2: return
    if len(size) != 2: return
    return self.openslide.read_region(loc, level, size)


  def get_total_image_by_level(self, seg_level:int=0):
    return self.openslide.read_region((0,0), seg_level, (self.width, self.height))


  def test(self):
    logging.debug("WSIMap class Test")
