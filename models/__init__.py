# Copyright (c) OpenMMLab. All rights reserved.
# from .aggregator import (Slot, ResultMap)
from .feature_extraction import (get_vgg_features, get_resnet_features, 
  get_image_features)

__all__ = [
  'get_vgg_features', 'get_resnet_features', 'get_image_features'
]