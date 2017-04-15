import numpy as np

from PIL import Image
from argparse import ArgumentParser

from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor

from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, SegNet, PSPNet, UNetSeg
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard

NUM_CHANNELS = 3
NUM_CLASSES = 22

def train(args, model, loader):
    model.train(True)

    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss2d()

    for epoch in range(1, args.num_epochs+1):
        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

def evaluate(args, model, loader):
    model.train(False)

    for step, (images, labels) in enumerate(loader):
        inputs = Variable(images)
        targets = Variable(labels)
        outputs = model(inputs)

        return outputs

def main(args):
    model = FCN8(NUM_CHANNELS, NUM_CLASSES)

    loader = DataLoader(VOC12(args.dataroot,
        input_transform=Compose([
            CenterCrop(256),
            ToTensor(),
        ]),
        target_transform=Compose([
            CenterCrop(256),
            ToLabel(),
            Relabel(255, 21),
        ])), num_workers=args.num_workers, batch_size=args.batch_size)

    if args.cuda:
        model = DataParallel(model).cuda()

    print(evaluate(args, model, loader).size())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['simple'])
    parser.add_argument('--visualize', choices=['dashboard'])
    parser.add_argument('--visualize-loss-steps', type=int, default=50)
    parser.add_argument('--visualize-image-steps', type=int, default=50)
    parser.add_argument('--num-epochs', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataroot', nargs='?', default='data')
    main(parser.parse_args())
