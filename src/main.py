import sys
sys.path.append("..")
import os
import time
from openslide import OpenSlide, OpenSlideError
from tqdm import tqdm
from aadw.utils.tissue_detection import segmentTissue
from aadw.utils.patch_generator import getPatchGenerator
from aadw.models.efficientNet import EfficientNet
import torch
from torchvision import transforms


filter_params={'a_t':10, 'a_h': 16, 'max_n_holes':5 }
patching_param = {'patch_level': 0, 
                  # 'patch_size': 1228, 
                  # 'step_size': 1228, 
                  'patch_size': 448, 
                  'step_size': 448, 
                  'save_path': 'test-patches', 
                  'custom_downsample': 2}


def get_list(dir_) -> list:
    list_ = []
#     for (dirpath, dirnames, filenames) in walk(dir_):
    for (_, _, filenames) in os.walk(dir_):
        list_ = filenames
    return list_


def main():
  if os.path.exists(patching_param['save_path']) is False:
      os.mkdir(patching_param['save_path'])
      
  dataset_path = 'dataset'
  model = EfficientNet.from_pretrained('efficientnet-b0')
  model.eval()

  heatmap = list()
  list_ = get_list(dataset_path)
  for file_ in list_:
    start = time.time()
    osr = OpenSlide(os.path.join(dataset_path, file_))
    end = time.time()
    print(f'[D] Open Whole Slide Image (Elapsed time): {end - start}s')
    start = time.time()
    ct, ht = segmentTissue(osr, filter_params=filter_params)  
    end = time.time()
    print(f'[D] Tissue Detection (Elapsed time): {end - start}s')

    filename = os.path.basename(file_).split('.')[0]

    tissue_map = list()
    for idx, cont in enumerate(ct):
  #     patch_gen = _getPatchGenerator(osr, cont, idx, ht, contour_fn='four_pt', **patching_param)
      start = time.time()
      patch_gen = getPatchGenerator(osr, cont, idx, ht, contour_fn='four_pt_hard', **patching_param)
      end = time.time()
      print(f'[D] Patch generation (Elapsed time): {end - start}s')
      try:
        first_patch = next(patch_gen)

      # empty contour, continue
      except StopIteration:
        print(f'StopIteration Error')
        continue
      
      for patch in patch_gen:

        print(f"[D] Patch: {filename}_x_{patch['x']}_y_{patch['y']}")
        # Prediction
        trans = transforms.ToTensor()
        tensor_data = trans(patch["patch_PIL"])
        tensor_data = tensor_data[None, :]
        start = time.time()
        outputs = model(tensor_data)
        end = time.time()
        print(f'[D] Single Prediction (Elapsed time): {end - start}s')
        
        patch_map = [patch['x'], patch['y'], torch.max(outputs)]
        tissue_map.append(patch_map)
        print(f'[D] prediction: {torch.max(outputs)}')


    osr.close()
    heatmap.append(tissue_map)

    print(f'[D] heatmap length: {len(heatmap)}')

  return 0


def test_main():
  model = EfficientNet.from_pretrained('efficientnet-b2')
  from torchsummary import summary
  summary(model.cuda(), (3, 224, 224))
  return 0

if __name__=="__main__":
  # main()
  test_main()