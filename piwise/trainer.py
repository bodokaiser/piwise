from torch.autograd import Variable

class Result:

    def __init__(self, epoch, step, input, output, target, loss):
        self.loss = loss
        self.step = step
        self.epoch = epoch
        self.input = input
        self.output = output
        self.target = target


class Trainer:

    def __init__(self, model, optimizer, criterion):
        self.is_cuda = False
        self.step_hooks = []
        self.epoch_hooks = []
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def cuda(self):
        self.is_cuda = True
        self.criterion = self.criterion.cuda()

    def plug(self, hook, every_n_steps=1, every_n_epochs=0):
        if every_n_steps > 0:
            self.step_hooks.append((every_n_steps, hook))
        if every_n_epochs > 0:
            self.epoch_hooks.append((every_n_epochs, hook))

    def train_epoch(self, loader, epoch=0):
        self.model.train()

        for step, (images, labels) in enumerate(loader):
            if self.is_cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets[:, 0])
            loss.backward()
            self.optimizer.step()

            result = Result(epoch, step, inputs[0], outputs[0], targets[0], loss)

            for n_step, hook in self.step_hooks:
                if step % n_step == 0:
                    hook(result)

        return result

    def train(self, loader, epochs):
        for epoch in range(1, epochs+1):
            result = self.train_epoch(loader, epoch)

            for n_epoch, hook in self.epoch_hooks:
                if epoch % n_epoch == 0:
                    hook(result)