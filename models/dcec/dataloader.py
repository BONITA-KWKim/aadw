import cv2 as cv
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.common import get_files

import os
from time import time
from models.feature_extraction import load_npy
class DCECBatchDataset(Dataset):
  def __init__(self, npy_dir, transform=None, img_size:list=[128, 128, 3]):
    self.npy_dir = npy_dir
    self.batchfiles = [ x for _, _, files in os.walk(npy_dir) for x in files \
      if x.endswith("npy") ]
    self.batch_length = len(self.batchfiles)
    with open(os.path.join(npy_dir, 'metadata'), 'rb') as f:
      self.length = int(f.read())
    # print(f'[D] length: {self.length}')

    # print(f'[D] batchfile name: {self.batchfiles}')

    self.batch_idx = 0
    t0 = time()
    self.cur_batch = load_npy(npy_dir, self.batchfiles[self.batch_idx])
    t1 = time()
    # print(f'[D] current batchfile name({round(t1-t0, 4)}s)\n\ttype: {type(self.cur_batch)}, len: {len(self.cur_batch)}')

    self.cur_length = len(self.cur_batch)
    self.offset = 0

    if transform==None:
      self.transform = transforms.Compose([
        transforms.Resize(img_size[0:2]),
      ])
    else:
      self.transform = transform

  def __len__(self):
    return self.length 

  def __getitem__(self, idx):
    if idx >= self.cur_length + self.offset:
      self.offset += self.cur_length
      self.batch_idx += 1
      self.cur_batch = load_npy(self.npy_dir, self.batchfiles[self.batch_idx])
      self.cur_length = len(self.cur_batch)
    
    img_info = self.cur_batch[idx-self.offset]
    image = img_info['image']
    image = self.transform(image)
    return image, img_info['path']


# class CustomImageDataset(Dataset):
class DCECDataset(Dataset):
  def __init__(self, img_dir, transform=None, img_size:list=[128, 128, 3]):
    self.images = get_files(img_dir, type="path")
    self.length = len(self.images)
    
    if transform==None:
      self.transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize(img_size[0:2]),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ])
    else:
      self.transform = transform

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    img_path = self.images[idx]
    image = cv.imread(img_path)
    image = self.transform(image)
    return image, img_path


from tqdm import tqdm
if __name__=="__main__":
  # train_data = DCECDataset('/data/kwkim/dataset/bladder/test_patches', img_size=[512, 512, 3])
  # dl = DataLoader(train_data, batch_size=1, shuffle=True)
  # train_batch = DCECBatchDataset('/data/kwkim/aadw/utils/batch-test001', img_size=[512, 512, 3])
  train_batch = DCECBatchDataset('/data/kwkim/aadw/utils/batch-test', img_size=[512, 512, 3])
  dl = DataLoader(train_batch, batch_size=8, shuffle=False)
  
  import torch
  device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")

  for data in tqdm(dl, desc='Dataloader Test ...'):
    data = data[0]
    inputs = data.to(device)
    # inputs = data
    # print(f'[D] data type: {inputs.type()}')
    # print(f'[D] data shape: {inputs.shape}')
    # print(f'[D] data: {inputs}')