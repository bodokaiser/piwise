import numpy as np
import torch

from PIL import Image

# can be used to convert from (grayscale) class to object label
def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r=r+(1<<(7-j))*((i&(1<<(3*j)))>>(3*j))
            g=g+(1<<(7-j))*((i&(1<<(3*j+1)))>>(3*j+1))
            b=b+(1<<(7-j))*((i&(1<<(3*j+2)))>>(3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()
