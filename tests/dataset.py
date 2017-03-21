import os
import unittest

from piwise import dataset

class TestVOC2012(unittest.TestCase):

    def setUp(self):
        self.dataset = dataset.VOC2012('data')
        self.dataset_len = sum([
            1 for f in os.listdir('data/labels') if f.endswith('.png')])

    def test_length(self):
        self.assertEqual(len(self.dataset), self.dataset_len)

    def test_getitem(self):
        image, label = self.dataset[0]

        self.assertEqual(image.mode, label.mode)
        self.assertEqual(image.width, label.width)
        self.assertEqual(image.height, label.height)
