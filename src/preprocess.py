import os
import argparse
from tqdm import tqdm
import umap # neads pyparsing==2.4.7

import utils.StainNorm as sn
from utils.common import get_files
from utils.common import create_output_directory
from models.feature_extraction import base_extractor 
from models.feature_extraction import get_a_image_feature_with_normailzation
from models.feature_extraction import get_a_image_feature
from models.feature_extraction import save_npy


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
  parser.add_argument("--umap", dest='umapflag', type=_str2bool, nargs='?',
                      const=True, default=False,
                      help="Umap Flag")
  args = parser.parse_args()
  return args


def reduce_dimension(feature_info, n_components:int=100):
  print(f'[I] UMAP')
  features = []
  for image_name in feature_info:
    features.append(feature_info[image_name])
  
  reducer = umap.UMAP(n_components=n_components)
  r = reducer.fit_transform(features)
  return r


NORMALIZATION_METHODS = {
  "he": sn.TYPE_HE,
  "eosin": sn.TYPE_EOSIN,
  "hematoxylin": sn.TYPE_HEMATOXYLIN
}
PREFIXES = {
  sn.TYPE_HE: "HE",
  sn.TYPE_EOSIN: "EOSIN",
  sn.TYPE_HEMATOXYLIN: "HEMATOXYLIN",
}
def create_feature_npy(images_info, featurename:str, model:str, normalization,   
                       saveflag, outdir, umapflag):
  # extract feature without normalization
  features = dict()
  new_model, device, transform = base_extractor(model)
  for item in tqdm(images_info, desc="Feature Extracting ..."):
    feature = get_a_image_feature(item["path"], new_model, device, transform)
    features[item["name"]] = feature
  save_npy(features, outdir, name=f'{featurename}-{model}')

  # reduct demension with UMAP
  if umapflag:
    reduced = reduce_dimension(features, n_components=100)
    save_npy(reduced, outdir, name=f'{featurename}-{model}-umap-100')
    reduced = reduce_dimension(features, n_components=60)
    save_npy(reduced, outdir, name=f'{featurename}-{model}-umap-60')
    reduced = reduce_dimension(features, n_components=20)
    save_npy(reduced, outdir, name=f'{featurename}-{model}-umap-20')
  
  # extract feature with normalization
  if "all"==normalization:
    methods = [
      NORMALIZATION_METHODS["he"],
      NORMALIZATION_METHODS["eosin"],
      NORMALIZATION_METHODS["hematoxylin"]
    ]
  elif "none"==normalization:
    methods = []
  else:
    methods = [NORMALIZATION_METHODS[normalization]]
 
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

    # reduct demension with UMAP
    if umapflag:
      reduced = reduce_dimension(stain_features, n_components=100)
      save_npy(reduced, outdir, name=f'{PREFIXES[method]}-{featurename}-{model}-umap-100')
      reduced = reduce_dimension(stain_features, n_components=60)
      save_npy(reduced, outdir, name=f'{PREFIXES[method]}-{featurename}-{model}-umap-60')
      reduced = reduce_dimension(stain_features, n_components=20)
      save_npy(reduced, outdir, name=f'{PREFIXES[method]}-{featurename}-{model}-umap-20')


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
        args.mode.lower(), args.saveflag, args.outdir, args.umapflag)

  return 0


'''
rm -r /data/kwkim/dataset/bladder/test_patches/features
python src/preprocess.py --input_dir /data/kwkim/dataset/bladder/test_patches \
--output_dir /data/kwkim/dataset/bladder/test_patches/features \
--feature_name features --mode all --model all --save False --umap True

python src/preprocess.py --input_dir /data/kwkim/dataset/bladder/test_patches \
--output_dir /data/kwkim/dataset/bladder/test_patches/features \
--feature_name features --mode he --model resnet --save False --umap True


python src/preprocess.py --input_dir /data/kwkim/dataset/bladder/trainset-v3.1 \
--output_dir /data/kwkim/dataset/bladder/trainset-v3.1/features \
--feature_name features --mode all --model all --save False --umap True

python src/preprocess.py --input_dir /data/kwkim/dataset/bladder/trainset_v3.0 \
--output_dir /data/kwkim/dataset/bladder/trainset_v3.0/features \
--feature_name features --mode all --model resnet --save False --umap True
'''
if __name__=='__main__':
  main()
