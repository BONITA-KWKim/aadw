import os
import cv2 as cv
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.common import get_files


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
    return image


if __name__=="__main__":
  train_data = DCECDataset('/data/kwkim/dataset/bladder/test_patches', img_size=[512, 512, 3])
  dl = DataLoader(train_data, batch_size=1, shuffle=True)
  
  import torch
  device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

  for data in dl:
    inputs = data.to(device)
    # inputs = data
    print(f'[D] data type: {inputs.type()}')
    print(f'[D] data shape: {inputs.shape}')
    # print(f'[D] data: {inputs}')