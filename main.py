import numpy as np

from PIL import Image
from argparse import ArgumentParser

from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage

from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet1, SegNet2
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard

NUM_CHANNELS = 3
NUM_CLASSES = 22

color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
])
target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])

def train(args, model):
    model.train(True)

    if args.cuda:
        model = DataParallel(model).cuda()

    loader = DataLoader(VOC12(args.datadir, input_transform, target_transform),
        num_workers=args.num_workers, batch_size=args.batch_size)

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

            if step % args.steps_loss == 0:
                print(f'epoch: {epoch}, step: {step}, loss: {loss.data[0]}')

def evaluate(args, model):
    model.train(False)

    image = input_transform(Image.open(args.image))
    label = color_transform(model(Variable(image).unsqueeze(0))[0].data.max(0)[1])

    image_transform(label).save(args.label)

def main(args):
    Net = None
    if args.model == 'fcn8':
        Net = FCN8
    if args.model == 'fcn16':
        Net = FCN16
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'unet':
        Net = UNet
    if args.model == 'pspnet':
        Net = PSPNet
    if args.model == 'segnet1':
        Net = SegNet1
    if args.model == 'segnet2':
        Net = SegNet2
    assert Net is not None, f'model {args.model} not available'

    model = Net(NUM_CHANNELS, NUM_CLASSES)

    if args.mode == 'eval':
        evaluate(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('image')
    parser_eval.add_argument('label')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=50)

    main(parser.parse_args())