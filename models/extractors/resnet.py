from torch import nn
from torchvision import models


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


def get_resnet_features():
  model = models.resnet152(pretrained=True)
  resnet_extractor = ResNetExtractor(model)
  return resnet_extractor