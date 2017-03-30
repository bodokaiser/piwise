import numpy as np

from torch.autograd import Variable

from visdom import Visdom

class Dashboard:

    def __init__(self, port):
        self.vis = Visdom(port=port)
        self.win = None

    def loss(self, losses, title):
        x = np.arange(1, len(losses)+1, 1)

        win = self.vis.line(losses, x, win=self.win, env='loss',
            opts=dict(title=title))

        if self.win is None:
            self.win = win

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy().transpose((1, 2, 0))

        self.vis.image(image, env='images', opts=dict(title=title))