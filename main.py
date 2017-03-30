import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from visdom import Visdom
from argparse import ArgumentParser

from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor

from piwise.dataset import Voc12
from piwise.network import Simple
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard

def train(args, model, loader, optimizer, criterion):
    model.train()

    if args.visualize:
        board = Dashboard(args.port)

    total_loss = []

    for epoch in range(args.num_epochs+1):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                criterion = criterion.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[0]).mul(-1)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            total_loss.append(loss.data[0])

            if args.visualize and step % args.visualize_steps == 0:
                colorize = Colorize()

                outputs = outputs[0].cpu().max(0)[1]
                targets = targets[0].cpu()

                output = colorize(outputs.data)
                output_median = outputs.view(-1).median(0)[0].data[0]
                target = colorize(targets.data)
                target_median = targets.view(-1).median(0)[0].data[0]

                if len(total_loss) > 1:
                    board.loss(total_loss, 'training loss')

                title = lambda n: f'{n} (epoch: {epoch}, step: {step})'

                board.image(inputs[0], title('input'))
                board.image(output, title(f'output [{output_median}]'))
                board.image(target, title(f'target [{target_median}]'))

        print(f'epoch: {epoch}, epoch_loss: {sum(epoch_loss)}')

NUM_CHANNELS = 3
NUM_CLASSES = 22

def main(args):
    if args.model == 'simple':
        model = Simple(NUM_CHANNELS, NUM_CLASSES)

    if args.cuda:
        model.cuda()

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

    train(args, model, loader, optimizer, criterion)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['basic', 'simple'], required=True)
    parser.add_argument('--visualize', choices=['dashboard'])
    parser.add_argument('--visualize-steps', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataroot', nargs='?', default='data')
    main(parser.parse_args())