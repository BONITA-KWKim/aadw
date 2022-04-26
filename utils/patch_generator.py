"""Example code
'''Patching params description
custom_downsample: 1 or 2
'''
from PIL import Image
patching_param = {'patch_level': 0, 
                  'patch_size': 512, 
                  'step_size': 512, 
                  'save_path': 'test_output', 
                  'custom_downsample': 1}
_counter = 0
for idx, cont in enumerate(ct):
    patch_gen = _getPatchGenerator(osr, cont, idx, ht, contour_fn='four_pt', **patching_param)
#     patch_gen = _getPatchGenerator(osr, cont, idx, ht, contour_fn='four_pt_hard', **patching_param)

    try:
        first_patch = next(patch_gen)

    # empty contour, continue
    except StopIteration:
        print(f'StopIteration Error')
        continue

    for patch in patch_gen:
        patch['patch_PIL'].save(f"{patch['save_path']}/TestPatch_{_counter}.png", "png")
#         Image.save(patch['save_path'] + "TestPatch" + _counter, "JPEG")
        _counter += 1        
#         with Image.open(patch['patch_PIL']) as im:
#             im.thumbnail(size)
#             im.save(patch['save_path'] + "TestPatch", _counter, "JPEG")
#             _couter += 1
"""
# other imports
import cv2
import numpy as np

class Contour_Checking_fn(object):
  # Defining __call__ method 
  def __call__(self, pt): 
    raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
  def __init__(self, contour):
    self.cont = contour

  def __call__(self, pt): 
    return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
  def __init__(self, contour, patch_size):
    self.cont = contour
    self.patch_size = patch_size

  def __call__(self, pt): 
    return 1 if cv2.pointPolygonTest(self.cont, (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2), False) >= 0 else 0

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
  def __init__(self, contour, patch_size, center_shift=0.5):
    self.cont = contour
    self.patch_size = patch_size
    self.shift = int(patch_size//2*center_shift)
  def __call__(self, pt): 
    center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
    if self.shift > 0:
      all_points = [(center[0]-self.shift, center[1]-self.shift),
              (center[0]+self.shift, center[1]+self.shift),
              (center[0]+self.shift, center[1]-self.shift),
              (center[0]-self.shift, center[1]+self.shift)
              ]
    else:
      all_points = [center]

    for points in all_points:
      if cv2.pointPolygonTest(self.cont, points, False) >= 0:
        return 1
    return 0

# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
  def __init__(self, contour, patch_size, center_shift=0.5):
    self.cont = contour
    self.patch_size = patch_size
    self.shift = int(patch_size//2*center_shift)
  def __call__(self, pt): 
    center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
    if self.shift > 0:
      all_points = [(center[0]-self.shift, center[1]-self.shift),
                    (center[0]+self.shift, center[1]+self.shift),
                    (center[0]+self.shift, center[1]-self.shift),
                    (center[0]-self.shift, center[1]+self.shift)]
    else:
      all_points = [center]

    for points in all_points:
      if cv2.pointPolygonTest(self.cont, points, False) < 0:
        return 0
    return 1

def isWhitePatch(patch, satThresh=5):
  patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
  return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
  return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isInHoles(holes, pt, patch_size):
  for hole in holes:
    if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
      return 1

  return 0


def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
  if cont_check_fn(pt):
    if holes is not None:
      return not isInHoles(holes, pt, patch_size)
    else:
      return 1
  return 0


def getPatchGenerator(wsi_img, cont, cont_idx, ht, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
  white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
  start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, wsi_img.level_dimensions[patch_level][0], wsi_img.level_dimensions[patch_level][1])
  print("Bounding Box:", start_x, start_y, w, h)
  print("Contour Area:", cv2.contourArea(cont))

  
  def _assertLevelDownsamples(wsi_img):
    level_downsamples = []
    dim_0 = wsi_img.level_dimensions[0]

    for downsample, dim in zip(wsi_img.level_downsamples, wsi_img.level_dimensions):
      estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
      level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples


  if custom_downsample > 1:
    assert custom_downsample == 2 
    target_patch_size = patch_size
    patch_size = target_patch_size * 2
    step_size = step_size * 2
    print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
      target_patch_size, target_patch_size))
  
  level_downsamples = _assertLevelDownsamples(wsi_img)
  patch_downsample = (int(level_downsamples[patch_level][0]), int(level_downsamples[patch_level][1]))
  ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])

  step_size_x = step_size * patch_downsample[0]
  step_size_y = step_size * patch_downsample[1]

  if isinstance(contour_fn, str):
    if contour_fn == 'four_pt':
      cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
    elif contour_fn == 'four_pt_hard':
      cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
    elif contour_fn == 'center':
      cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
    elif contour_fn == 'basic':
      cont_check_fn = isInContourV1(contour=cont)
    else:
      raise NotImplementedError
  else:
    assert isinstance(contour_fn, Contour_Checking_fn)
    cont_check_fn = contour_fn

  img_w, img_h = wsi_img.level_dimensions[0]
  if use_padding:
    stop_y = start_y+h
    stop_x = start_x+w
  else:
    stop_y = min(start_y+h, img_h-ref_patch_size[1])
    stop_x = min(start_x+w, img_w-ref_patch_size[0])

  count = 0
  for y in range(start_y, stop_y, step_size_y):
    for x in range(start_x, stop_x, step_size_x):

      if not isInContours(cont_check_fn, (x,y), ht[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
        continue  

      count+=1
      patch_PIL = wsi_img.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
      if custom_downsample > 1:
        patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

      if white_black:
        if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
          continue

      patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 
                    'y':y // (patch_downsample[1] * custom_downsample), 
                    'cont_idx':cont_idx, 'patch_level':patch_level, 
                    'downsample': level_downsamples[patch_level], 
                    'downsampled_level_dim': tuple(np.array(wsi_img.level_dimensions[patch_level])//custom_downsample), 
                    'level_dim': wsi_img.level_dimensions[patch_level],
                    'patch_PIL':patch_PIL, 
                    #'name':self.name, 
                    'save_path':save_path}

      yield patch_info