import torch
import cv2
import numpy as np
from torch import nn
from torchvision import models, transforms
from torchsummary import summary


class VggExtractor(nn.Module):
  def __init__(self, model):
    super(VggExtractor, self).__init__()
    # Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    # Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
    # Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
    # Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
    # It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 


class ResNetExtractor(nn.Module):
  def __init__(self, model):
    super(ResNetExtractor, self).__init__()
  
    self.conv1 = model.conv1
    self.bn1 = model.bn1
    self.relu = model.relu
    self.maxpool = model.maxpool
    self.layer1 = list(model.layer1)
    self.layer1 = nn.Sequential(*self.layer1)
    self.layer2 = list(model.layer2)
    self.layer2 = nn.Sequential(*self.layer2)
    self.layer3 = list(model.layer3)
    self.layer3 = nn.Sequential(*self.layer3)
    self.layer4 = list(model.layer4)
    self.layer4 = nn.Sequential(*self.layer4)
    self.avgpool = model.avgpool
    self.flatten = nn.Flatten()

  def forward(self, x):
    # It will take the input 'x' until it returns the feature vector called 'out'
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out) 
    out = self.flatten(out)
    return out 


def get_vgg_features():
  model = models.vgg16(pretrained=True)
  vgg_extractor = VggExtractor(model)
  return vgg_extractor
  
def get_resnet_features():
  model = models.resnet152(pretrained=True)
  resnet_extractor = ResNetExtractor(model)
  return resnet_extractor


def get_model(type:str="resnet"):
  if type == "resnet":
    new_model = get_resnet_features()
  elif type == "vgg":
    new_model = get_vgg_features()
  else:
    new_model = get_resnet_features()

  # Change the device to GPU
  device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
  new_model = new_model.to(device)
  return new_model, device

def get_transform():
  # Transform the image, so it becomes readable with the model
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()                
  ])
  return transform

def get_image_features(samples, type:str="resnet"):
  new_model, device = get_model(type)
  transform = get_transform()
  features = []
  for item in samples:
    img = cv2.imread(item)
    img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    
    with torch.no_grad():
      feature = new_model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
    
    features.append(feature.cpu().detach().numpy().reshape(-1))

  # Convert to NumPy Array
  features = np.array(features)
  return features



if __name__=="__main__":
  # new_model = get_resnet_features()
  # summary(new_model, (3, 512, 512), device='cpu')
  new_model = get_vgg_features()
  summary(new_model, (3, 224, 224), device='cpu')
