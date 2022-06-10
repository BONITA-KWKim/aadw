import os
import argparse
import cv2
from tqdm import tqdm

import utils.StainNorm as sn
from utils.common import get_files
from utils.common import create_output_directory
from models.feature_extraction import base_extractor 
from models.feature_extraction import get_a_image_feature_with_normailzation
from models.feature_extraction import get_a_image_feature
from models.feature_extraction import save_npy
from models.feature_extraction import load_npy


def arguments():
  
  def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser(description='Image clustering')
  parser.add_argument('--mode', dest='mode', choices=['none', 'he', 'eosin', 'hematoxylin', 'all'],
                      default='HE', 
                      help='Operation mode [none|HE|EOSIN|HEMATOXYLIN|ALL] (default: none)')
  parser.add_argument('--model', dest='model', choices=['vgg', 'resnet', 'nasnet', 'all'],
                      default='resnet',
                      help='Neural network architecture [vgg|resnet|nasnet|all] (default: resnet)')
  parser.add_argument('--feature_name', dest='featurename', 
                      default='features',
                      help='Feature Name (default: features)')
  parser.add_argument("--save", dest='saveflag', type=_str2bool, nargs='?',
                      const=True, default=False,
                      help="Save Flag")
  parser.add_argument('--input_dir', dest='indir', 
                      default='./input',
                      help='Train Dataset Directory (default: ./input)')
  parser.add_argument('--output_dir', dest='outdir', 
                      default='./output', 
                      help='Result Directory (default: ./output)')
  args = parser.parse_args()
  return args


PREFIXES = {
  sn.TYPE_HE: "HE",
  sn.TYPE_EOSIN: "EOSIN",
  sn.TYPE_HEMATOXYLIN: "HEMATOXYLIN",
}
def create_feature_npy(images_info, featurename:str, model:str, normalization,   
                       saveflag, outdir):
  features = dict()
  new_model, device, transform = base_extractor(model)
  for item in tqdm(images_info, desc="Feature Extracting ..."):
    feature = get_a_image_feature(item["path"], new_model, device, transform)
    features[item["name"]] = feature
  save_npy(features, outdir, name=f'{featurename}-{model}')

  if "none"==normalization: return
  
  if "he"==normalization:
    methods = [sn.TYPE_HE]
  elif "eosin"==normalization:
    methods = [sn.TYPE_EOSIN]
  elif "hematoxylin"==normalization:
    methods = [sn.TYPE_HEMATOXYLIN]
  elif "all"==normalization:
    methods = [sn.TYPE_HE, sn.TYPE_EOSIN, sn.TYPE_HEMATOXYLIN]
 
  for method in methods:
    print(f'[D] prefix: {PREFIXES[method]}')
    stain_features = dict()
    stain_outdir = os.path.join(outdir, PREFIXES[method])
    if saveflag:
      create_output_directory(stain_outdir)
    for item in tqdm(images_info, desc="Stain Normalization ..."):
      feature = get_a_image_feature_with_normailzation(item, method, saveflag, 
                    stain_outdir, PREFIXES[method], new_model, device, transform)
      stain_features[item["name"]] = feature
    save_npy(stain_features, outdir, name=f'{PREFIXES[method]}-{featurename}-{model}')


def main():
  args = arguments()
  create_output_directory(args.outdir)
  
  images_info = get_files(args.indir, type="both")
  if "all"==args.model:
    models =["vgg", "resnet", "nasnet"]
  else:
    models = [args.model]

  for model in models:
    create_feature_npy(images_info, args.featurename, model, 
        args.mode.lower(), args.saveflag, args.outdir)

  return 0


'''
rm -r /data/kwkim/aadw/stain_test
python src/stain_normalization.py --input_dir /data/kwkim/dataset/bladder/trainset_v3.0 \
--output_dir /data/kwkim/dataset/bladder/trainset_v3.0 \
--feature_name features --mode all --model resnet --save False


python src/stain_normalization.py --input_dir /data/kwkim/dataset/bladder/test_patches \
--output_dir /data/kwkim/dataset/bladder/test_patches/features \
--feature_name features --mode all --model all --save False
'''
if __name__=='__main__':
  main()
