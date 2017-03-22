import argparse
import numpy as np
import torch
import visdom

from piwise import network, dataset, transform

from torch import nn, optim, autograd
from torch.utils import data
from torchvision import utils, transforms

NUM_CLASSES = 22

def vis_loss(vis, win, losses):
    opts = dict(title='training loss')

    return vis.line(losses, np.arange(1, len(losses)+1, 1), win=win,
        env='loss', opts=opts)

def vis_image(vis, image, epoch, step, name):
    if not isinstance(image, np.ndarray):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, autograd.Variable):
            image = image.data
        image = image.numpy()

    if len(image.shape) == 3:
        if image.shape[0] == 3:
            image = image.transpose((1, 2, 0))
        if image.shape[0] == 1:
            image = image[0].transpose((0, 1))
    if image.dtype == np.int64:
        image = image.astype(np.uint8)

    opts = dict(title=f'{name} (epoch: {epoch}, step: {step})')

    vis.heatmap(np.flipud(image), env='images', opts=opts)

def train(args, model, loader, optimizer, criterion):
    model.train()

    if args.visualize:
        win = None
        vis = visdom.Visdom(port=args.port)

    total_loss = []

    for epoch in range(args.num_epochs+1):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                criterion = criterion.cuda()

            inputs = autograd.Variable(images)
            targets = autograd.Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            total_loss.append(loss.data[0])

            if args.visualize and step % args.visualize_steps == 0:
                if len(total_loss) > 1:
                    win = vis_loss(vis, win, total_loss)

                output = outputs[0]
                if output.is_cuda:
                    output = output.cpu()
                output = output.data.numpy().argmax(0)

                vis_image(vis, inputs[0], epoch, step, 'input')
                vis_image(vis, output, epoch, step, 'output')
                vis_image(vis, targets[0], epoch, step, 'target')

        print(f'epoch: {epoch}, epoch_loss: {sum(epoch_loss)}')


def main(args):
    if args.model == 'basic':
        model = network.Basic(NUM_CLASSES)
    if args.model == 'unet':
        model = network.UNet(NUM_CLASSES)

    if args.cuda:
        model.cuda()

    loader = data.DataLoader(dataset.VOC2012(args.dataroot,
        input_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # our models currently only support one color channel 
            transforms.Lambda(lambda t: t[0].unsqueeze(0)),
        ]),
        target_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transform.ToLabel(),
            # reduces number of classes to 22 as we drop 22-255
            transform.Relabel(255, 21),
        ])), num_workers=args.num_workers, batch_size=args.batch_size)

    optimizer = optim.Adam(model.parameters())
    criterion = network.CrossEntropy2d()

    train(args, model, loader, optimizer, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', choices=['basic', 'unet'], required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize-steps', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataroot', nargs='?', default='data')
    main(parser.parse_args())
