import argparse

from piwise import network, dataset, trainer

from torch import nn, optim
from torch.utils import data
from torchvision import transforms

def main(args):
    model = network.Basic()

    if args.cuda:
        model.cuda()

    criterion = network.CrossEntropySoftmax2d()

    dataloader = data.DataLoader(dataset.VOC2012(args.dataroot,
        input_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]),
        target_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mul(255).long()[0]),
        ])), shuffle=True, num_workers=args.num_workers)

    trainer.Trainer(model, criterion, dataloader).run(args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--dataroot', nargs='?', default='data')
    parser.add_argument('--trainroot', nargs='?', default='train')
    main(parser.parse_args())
