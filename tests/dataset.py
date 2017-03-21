import os
import unittest

from piwise import dataset

class TestVOC2012(unittest.TestCase):

    def setUp(self):
        self.dataset = dataset.VOC2012('data')
        self.dataset_len = sum([
            1 for f in os.listdir('data/classes') if f.endswith('.png')])

    def test_length(self):
        self.assertEqual(len(self.dataset), self.dataset_len)

    def test_getitem(self):
        image, classes = self.dataset[0]

        self.assertEqual(image.mode, classes.mode)
        self.assertTupleEqual(image.size, classes.size)
