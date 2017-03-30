import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.NLLLoss2d()

    def forward(self, x, y):
        return self.loss(x, y).mul(-1)