import sys
sys.path.append("..")
import torch
from aadw.models.efficientNet import EfficientNet


def prediction():
  inputs = torch.rand(1, 3, 224, 224)
  model = EfficientNet.from_pretrained('efficientnet-b0')
  model.eval()
  outputs = model(inputs)
  return outputs


if __name__=="__main__":
  prediction()