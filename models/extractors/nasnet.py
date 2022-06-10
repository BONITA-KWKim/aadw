import timm
from torch import nn


class NasnetLargeExtractor(nn.Module):
  def __init__(self, model):
    super(NasnetLargeExtractor, self).__init__()
    
    self.conv0 = model.conv0
    self.cell_stem_0 = model.cell_stem_0
    self.cell_stem_1 = model.cell_stem_1
    self.cell_0 = model.cell_0
    self.cell_1 = model.cell_1
    self.cell_2 = model.cell_2
    self.cell_3 = model.cell_3
    self.cell_4 = model.cell_4
    self.cell_5 = model.cell_5
    self.reduction_cell_0 = model.reduction_cell_0
    self.cell_6 = model.cell_6
    self.cell_7 = model.cell_7
    self.cell_8 = model.cell_8
    self.cell_9 = model.cell_9
    self.cell_10 = model.cell_10
    self.cell_11 = model.cell_11
    self.reduction_cell_1 = model.reduction_cell_1
    self.cell_12 = model.cell_12
    self.cell_13 = model.cell_13
    self.cell_14 = model.cell_14
    self.cell_15 = model.cell_15
    self.cell_16 = model.cell_16
    self.cell_17 = model.cell_17
    self.act = model.act
    self.global_pool = model.global_pool

  def forward(self, x):
    out = self.conv0(x)
    out = self.cell_stem_0(out)
    out = self.cell_stem_1(out)
    out = self.cell_0(out)
    out = self.cell_1(out)
    out = self.cell_2(out)
    out = self.cell_3(out)
    out = self.cell_4(out)
    out = self.cell_5(out)
    out = self.reduction_cell_0(out)
    out = self.cell_6(out)
    out = self.cell_7(out)
    out = self.cell_8(out)
    out = self.cell_9(out)
    out = self.cell_10(out)
    out = self.cell_11(out)
    out = self.reduction_cell_1(out)
    out = self.cell_12(out)
    out = self.cell_13(out)
    out = self.cell_14(out)
    out = self.cell_15(out)
    out = self.cell_16(out)
    out = self.cell_17(out)
    out = self.act(out)
    out = self.global_pool(out)


def get_nasnetlarge_features():
  model = timm.create_model('nasnetalarge', pretrained=True)
  extractor = NasnetLargeExtractor(model)
  return extractor
  

