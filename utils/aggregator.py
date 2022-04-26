MAJOR_VERSION = '0'
MINOR_VERSION = '1'
__version__ = f'v{MAJOR_VERSION}.{MINOR_VERSION}'


class Slot:
  def __init__(self, st_x, st_y, ds, raw, elected):
    self.start_x = st_x
    self.start_y = st_y
    self.downsampling = ds
    self.raw = raw
    self.elected = elected

  def __version__(self):
    return __version__

  def test(self):
    print("Slot Test")


class ResultMap:
  def __init__(self):
    self.__result_map = list()

  def __version__(self):
    return __version__

  def test(self):
    print("ResultMap Test")

  def add_slot(self, slot: Slot):
    self.__result_map.append(slot)
  
  def len(self):
    return len(self.__result_map)

