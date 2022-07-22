import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchsummary import summary
import utils.StainNorm as sn

from models.extractors.vgg import get_vgg_features
from models.extractors.resnet import get_resnet_features
from models.extractors.nasnet import get_nasnetlarge_features


def get_model(type_:str="resnet"):
  if type_ == "resnet":
    model = get_resnet_features()
  elif type_ == "vgg":
    model = get_vgg_features()
  elif type_ == "nasnetlarge":
    model = get_nasnetlarge_features()
  else:
    model = get_resnet_features()

  # Change the device to GPU
  device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  return model, device

def get_transform():
  # Transform the image, so it becomes readable with the model
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()                
  ])
  return transform


def base_extractor(type_:str="resnet"):
  model, device = get_model(type_)
  transform = get_transform()
  return model, device, transform


def get_image_features(samples, model, device, transform):
  features = []
  for item in samples:
    img = cv2.imread(item)
    img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    
    with torch.no_grad():
      feature = model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
    
    features.append(feature.cpu().detach().numpy().reshape(-1))

  # Convert to NumPy Array
  features = np.array(features)
  return features


def get_a_image_feature(file_, model, device, transform):
  img = cv2.imread(file_)
  img = transform(img)
  # Reshape the image. PyTorch model reads 4-dimensional tensor
  # [batch_size, channels, width, height]
  img = img.reshape(1, 3, 448, 448)
  img = img.to(device)
  
  with torch.no_grad():
    feature = model(img)
  return feature.cpu().detach().numpy().reshape(-1)


def get_a_image_feature_with_normailzation(item, mode, saveflag, outdir, prefix, model, device, transform):
  img = cv2.imread(item["path"])
  stained = sn.getStainImage(mode, img, dosave=saveflag, 
      saveFile=os.path.join(outdir, prefix + item["name"]))
  img = transform(stained)
  # Reshape the image. PyTorch model reads 4-dimensional tensor
  # [batch_size, channels, width, height]
  img = img.reshape(1, 3, 448, 448)
  img = img.to(device)
  
  with torch.no_grad():
    feature = model(img)
  return feature.cpu().detach().numpy().reshape(-1)


def save_npy(features, dir_, name:str="features"):
  npy_filename = os.path.join(dir_, name)
  np.save(npy_filename, features) # x_save.npy


def load_npy(dir_: str, name:str="features"):
  npy = None
  if name.endswith("npy"):
    filename = os.path.join(dir_, f'{name}')
  else:
    filename = os.path.join(dir_, f'{name}.npy')

  # print(f'[D] feature filename: {filename}')
  if os.path.exists(filename):
    npy = np.load(filename, allow_pickle=True)
    # print(f'[D] npy type: type({type(npy)})')
    npy = npy if isinstance(npy, np.ndarray) else npy.item()
    # npy = npy.item()
  return npy


if __name__=="__main__":
  new_model = get_resnet_features()
  summary(new_model, (3, 512, 512), device='cpu')
  new_model = get_vgg_features()
  summary(new_model, (3, 224, 224), device='cpu')

  # model = timm.create_model('nasnetalarge', pretrained=True)
  # summary(model, (3, 448, 448), device='cpu')
  new_model, _ = get_model(type_="nasnetlarge")
  print(new_model)
