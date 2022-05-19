import sys
sys.path.append("..")
import os

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import torchvision.transforms.functional as TF
import torch
import numpy as np
from PIL import Image

def random_hflip(image):
    image = TF.hflip(image)
    return image
   

def get_transform():
  # Transform the image, so it becomes readable with the model
  transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.CenterCrop(512),
    # transforms.Resize(448),
    transforms.ToTensor()                
  ])
  return transform


def get_list(dir_: str) -> list:
    return [os.path.join(d, x) for d, _, files in os.walk(dir_) for x in files]

transform = get_transform()
samples = get_list("/data/kwkim/dataset/bladder/oneshot_raw")


transform_h = transforms.Compose([
  transforms.ToPILImage(),
  # transforms.CenterCrop(512),
  # transforms.Resize(448),
  # transforms.ToTensor()                
])
for sample in samples:
  print(f'filename: {sample}')

  if not os.path.isdir(f'Images/{os.path.basename(sample).split(".")[0]}'):
    print('[D]The directory is not present. Creating a new one..')
    os.makedirs(f'Images/{os.path.basename(sample).split(".")[0]}')

  img = cv2.imread(sample)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = transform(img)
  
  print('crop')
  for i in range(5):
    aug_f = transforms.RandomResizedCrop((448, 448))
    crop = aug_f(img)
    crop.numpy()
    save_image(crop, f'Images/{os.path.basename(sample).split(".")[0]}/crop_{i+1}.{os.path.basename(sample).split(".")[1]}')
  
  img = transforms.ToPILImage()(img)
  img = transforms.Resize(448)(img)
  # img_h = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  print('hflip')
  # im_pil = Image.fromarray(img)
  im_pil = TF.hflip(img)
  im = transforms.ToTensor()(im_pil)
  save_image(im, f'Images/{os.path.basename(sample).split(".")[0]}/hflip.{os.path.basename(sample).split(".")[1]}')
  print('vflip')
  # im_pil = Image.fromarray(img)
  im_pil = TF.vflip(img)
  im = transforms.ToTensor()(im_pil)
  save_image(im, f'Images/{os.path.basename(sample).split(".")[0]}/vflip.{os.path.basename(sample).split(".")[1]}')
  

  img = transforms.ToTensor()(img)
  img = img.numpy()
  print('Gaussian Noise 001')
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.005, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/gaussian_1.{os.path.basename(sample).split(".")[1]}')
  print('Gaussian Noise 002')
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/gaussian_2.{os.path.basename(sample).split(".")[1]}')
  print('Gaussian Noise 003')
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.02, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/gaussian_3.{os.path.basename(sample).split(".")[1]}')
  print('Gaussian Noise 004')
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.03, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/gaussian_4.{os.path.basename(sample).split(".")[1]}')

  print('S&P 001')
  s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.1, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/sp_1.{os.path.basename(sample).split(".")[1]}')
  print('S&P 002')
  s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.5, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/sp_2.{os.path.basename(sample).split(".")[1]}')
  print('S&P 003')
  s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.7, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/sp_3.{os.path.basename(sample).split(".")[1]}')
  print('S&P 004')
  s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.9, clip=True))
  save_image(gauss_img, f'Images/{os.path.basename(sample).split(".")[0]}/sp_4.{os.path.basename(sample).split(".")[1]}')
