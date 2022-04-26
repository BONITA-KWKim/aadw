import sys
sys.path.append("..")
import unittest
from aadw.utils.aggregator import ResultMap
from aadw.utils.aggregator import Slot

class TestAggregator(unittest.TestCase):

  # def test_upper(self):
  #   self.assertEqual('foo'.upper(), 'FOO')

  # def test_isupper(self):
  #   self.assertTrue('FOO'.isupper())
  #   self.assertFalse('Foo'.isupper())

  # def test_split(self):
  #   s = 'hello world'
  #   self.assertEqual(s.split(), ['hello', 'world'])
  #   # check that s.split fails when the separator is not a string
  #   with self.assertRaises(TypeError):
  #       s.split(2)
  def __init__(self, *args, **kwargs):
    super(TestAggregator, self).__init__(*args, **kwargs)
    self.r = ResultMap()
    self.s1 = Slot(0, 0, 1, [0, 0, 1], 0)
    self.s2 = Slot(0, 0, 1, [0, 1, 1], 1)
      
  def test_result_map(self):
    self.r.test()

  def test_result_map_version(self):
    self.assertEqual(self.r.__version__(), 'v0.1')

  def test_result_map_add(self):
    self.r.add_slot(self.s1)
    self.r.add_slot(self.s2)
    self.assertEqual(self.r.len(), 2)

  def test_slot(self):
    self.s1.test()

  def test_slot_version(self):
    self.assertEqual(self.s1.__version__(), 'v0.1')

if __name__ == '__main__':
  unittest.main()