import numpy as np

from PIL import Image
from argparse import ArgumentParser

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor

from piwise.dataset import Voc12
from piwise.trainer import Trainer
from piwise.network import SimpleCNN, AdvancedCNN, UNet
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard

class LossHook:

    def __init__(self, interval, board=None):
        self.board =  board
        self.interval = interval

    def __call__(self, result):
        if result.step == 0:
            self.loss = []
        if result.step % self.interval == 0:
            self.loss.append(result.loss.data[0])

            if len(self.loss) > 1:
                if self.board is not None:
                    self.board.loss(self.loss, 'training loss')

            mean = np.mean(self.loss)
            print(f'epoch: {result.epoch}, step: {result.step}, loss: {mean}')

class ImageHook:

    def __init__(self, board):
        self.board = board

    def __call__(self, result):
        colorize = Colorize()

        output = result.output.cpu().max(0)[1]
        target = result.target.cpu()

        output_class = output.view(-1).median(0)[0].data[0]
        target_class = target.view(-1).median(0)[0].data[0]

        self._visualize(result, result.input, 'input')
        self._visualize(result, colorize(output.data),
            f'output [{output_class}]')
        self._visualize(result, colorize(target.data),
            f'target [{target_class}]')

    def _visualize(self, result, image, name):
        title = f'{name} (epoch: {result.epoch}, step: {result.step})'
        self.board.image(image, title)

NUM_CHANNELS = 3
NUM_CLASSES = 22

def main(args):
    Net = SimpleCNN

    if args.model == 'advanced':
        Net = AdvancedCNN
    if args.model == 'unet':
        Net = UNet

    model = Net(NUM_CHANNELS, NUM_CLASSES)

    loader = DataLoader(Voc12(args.dataroot,
        input_transform=Compose([
            CenterCrop(256),
            ToTensor(),
        ]),
        target_transform=Compose([
            CenterCrop(256),
            ToLabel(),
            Relabel(255, 21),
        ])), num_workers=args.num_workers, batch_size=args.batch_size)

    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss2d()
    trainer = Trainer(model, optimizer, criterion)

    if args.cuda:
        model.cuda()
        trainer.cuda()
    if args.visualize:
        board = Dashboard(args.port)

    trainer.plug(LossHook(args.visualize_loss_steps, board))
    trainer.plug(ImageHook(board), args.visualize_image_steps)
    trainer.train(loader, args.num_epochs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['simple', 'advanced', 'unet'])
    parser.add_argument('--visualize', choices=['dashboard'])
    parser.add_argument('--visualize-loss-steps', type=int, default=50)
    parser.add_argument('--visualize-image-steps', type=int, default=50)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataroot', nargs='?', default='data')
    main(parser.parse_args())