import os
import numpy as np
import cv2 as cv

from time import time
from absl import app
from absl import flags
from absl import logging
from torchvision import transforms

from utils.common import get_files
from models.feature_extraction import save_npy


FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, "Your name.")
# flags.DEFINE_string('imgpath', './img', 'Image directory')
flags.DEFINE_string('imgpath', None, 'Image directory')
flags.DEFINE_string('batchdir', None, 'Save directory for Batch files')
flags.DEFINE_string('batchname', None, 'Batch file name')
flags.DEFINE_enum('verbose', 'debug', ['info', 'debug', 'warning', 'error', 'fatal'], 'Verbose')
# Required flag.
flags.mark_flag_as_required("imgpath")
flags.mark_flag_as_required("batchdir")
flags.mark_flag_as_required("batchname")

VERBOSE = {
  "info": logging.INFO,
  "debug": logging.DEBUG,
  "warning": logging.WARNING,
  "error": logging.ERROR,
  "fatal": logging.FATAL
}

def split_samples(list_:list, div_line:int=1000) -> list:
  if div_line < len(list_):
    n, _ = divmod(len(list_), div_line)
    split_list = np.array_split(list_, n)
  else:
    split_list = np.array_split(list_, 1)
  return split_list


def timestamp(t0, t1):
  return round(t1-t0, 4)


''' Usage
python image_batch.py --imgpath /data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1 \
--batchdir /data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1/batch \
--batchname patchv4_1-train

nohup python image_batch.py --imgpath /data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1 \
--batchdir /data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1/batch \
--batchname patchv4_1-train > batch-20220721.log &

python image_batch.py --imgpath /data/kwkim/dataset/bladder/dcec_testset \
--batchdir /data/kwkim/aadw/utils/batch-test001 \
--batchname batch-test
'''
def main(argv):
  del argv # unused

  logging.set_verbosity(VERBOSE[FLAGS.verbose])

  # logging.debug('verbose test')
  # logging.info('verbose test')
  # logging.warning('verbose test')
  # logging.error('verbose test')

  files = get_files(FLAGS.imgpath, type="both")
  # logging.debug(f'file length: {len(files)}')
  # if len(files)>0: logging.debug(f'sample: {files[0]}')

  transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
  ])

  if not os.path.isdir(FLAGS.batchdir):
    logging.debug('The directory is not present. Creating a new one..')
    os.makedirs(FLAGS.batchdir)
  else:
    logging.debug('The directory is present.')

  with open(os.path.join(FLAGS.batchdir, 'metadata'), 'w') as f:
    f.write(str(len(files)))

  splits = split_samples(files, div_line=1000)
  logging.debug(f'Split list length: {len(splits)}')
  for idx, split in enumerate(splits):
    batch_list = list()
    t0 = time()
    for f in split:
      img = cv.imread(f['path'])
      img = transform(img)
      batch_list.append({'name': f['name'], 'path': f['path'], 'image': img})
    t1 = time()

    logging.debug(f'batch length({timestamp(t0, t1)}s): {len(batch_list)}')

    # save a batch file    
    save_npy(batch_list, FLAGS.batchdir, name=f'{FLAGS.batchname}_{idx}')
    logging.debug(f'save batch file: {f"{FLAGS.batchdir}/{FLAGS.batchname}_{idx}.npy"}')
  

if __name__=="__main__":
  app.run(main)
