import argparse

from piwise import network, dataset

from torch import nn, optim, autograd
from torch.utils import data
from torchvision import transforms

def main(args):
    model = network.Basic()

    if args.cuda:
        model.cuda()

    loader = data.DataLoader(dataset.VOC2012(args.dataroot,
        input_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]),
        target_transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mul(255).long()[0]),
        ])), shuffle=True, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters())
    criterion = network.CrossEntropySoftmax2d()

    for epoch in range(args.num_epochs+1):
        epoch_loss = 0

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            inputs = autograd.Variable(images)
            targets = autograd.Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0]

        print(f'epoch: {epoch}, loss: {epoch_loss}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--dataroot', nargs='?', default='data')
    parser.add_argument('--trainroot', nargs='?', default='train')
    main(parser.parse_args())
