from torch import nn, optim
from torch.utils import trainer
from torch.utils.trainer import plugins

class Trainer(trainer.Trainer):

    def __init__(self, model, criterion, dataset):
        optimizer = optim.Adam(model.parameters())

        super().__init__(model, criterion, optimizer, dataset)
        #super().register_plugin(plugins.LossMonitor())
        super().register_plugin(plugins.ProgressMonitor())
