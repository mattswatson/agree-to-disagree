"""
Model for classification task on MNIST
https://github.com/pytorch/examples/tree/master/mnist
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
from torchcontrib.optim import SWA
import pandas as pd
import numpy as np
import os
import pickle
import pickletools

from GaborNet import GaborConv2d

from utils import AverageMeter, VisdomLinePlotter


class Net(nn.Module):
    def __init__(self, dropout_prob1=0.25, dropout_prob2=0.25):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_prob1)
        self.dropout2 = nn.Dropout2d(dropout_prob2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output


class SmallNet(nn.Module):
    def __init__(self, dropout_prob1=0.25, softmax=True):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_prob1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5408, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.do_softmax = softmax

    def forward(self, x, features=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x1 = torch.flatten(x, 1)
        x = self.fc1(x1)

        if self.do_softmax:
            return self.softmax(x)

        if not features:
            return x

        return x, x1


class GaborNet(nn.Module):
    def __init__(self, dropout_prob1=0.25, softmax=True):
        super(GaborNet, self).__init__()

        self.g1 = GaborConv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1)
        self.dropout1 = nn.Dropout2d(dropout_prob1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(5408, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.do_softmax = softmax

    def forward(self, x, features=False):
        x = self.g1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x1 = torch.flatten(x, 1)
        x = self.fc1(x1)

        if self.do_softmax:
            return self.softmax(x)

        if not features:
            return x

        return x, x1


class MLP(nn.Module):
    def __init__(self, hidden_1_size=412, hidden_2_size=512, dropout_prob=0.25):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, 10)
        self.droput = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten the MNIST image input
        x = x.view(-1, 28 * 28)

        x = self.relu(self.fc1(x))

        x = self.droput(x)
        x = self.relu(self.fc2(x))

        x = self.droput(x)

        x = self.fc3(x)

        output = self.softmax(x)
        return output


def train(args, model, device, train_loader, optimizer, loss_fn, epoch, verbose=False, swa=None, param_path=None):
    losses = AverageMeter()

    model.train()
    len_train = len(train_loader.dataset) if len(train_loader.dataset) <= len(train_loader.sampler) \
        else len(train_loader.sampler)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Make sure we have the correct data type
        data = data.type(torch.FloatTensor)

        data, target = data.to(device), target.to(device)

        if swa is None:
            optimizer.zero_grad()
        else:
            swa.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        losses.update(loss.item(), len(target))

        if swa is None:
            optimizer.step()
        else:
            swa.step()

        if param_path is not None:
            state_dict = model.state_dict()

            save_path = os.path.join(param_path, 'epoch{}'.format(epoch))
            os.makedirs(save_path, exist_ok=True)

            save_path = os.path.join(save_path, 'batch{}.pkl'.format(batch_idx))

            with open(save_path, 'wb') as f:
                pickle.dump(state_dict, f)

            if verbose:
                print('======== Saved params for epoch {}, batch {} to {}'.format(epoch, batch_idx, save_path))

        if batch_idx % args.log_interval == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len_train,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    if swa is not None:
        swa.swap_swa_sgd()


    return losses.avg


def test(model, device, test_loader, loss_fn, resnet=False, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Make sure we have the correct data type
            data = data.type(torch.FloatTensor)

            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss

            if resnet:
                output = torch.softmax(output, dim=1)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(data)


    test_loss /= num_samples

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     num_samples,
                                                                                     100. * correct /
                                                                                    num_samples))
    acc = 100. * correct / num_samples
    return test_loss, acc


def load_model(path, model_type, state_dict, device):
    if state_dict:
        if model_type == 'large-cnn':
            model = Net(dropout_prob1=0.25, dropout_prob2=0.25).to(device)
        elif model_type == 'small-cnn':
            model = SmallNet(dropout_prob1=0.25).to(device)
        elif model_type == 'mlp':
            model = MLP(dropout_prob=0.25).to(device)
        elif model_type == 'gabor':
            model = GaborNet(dropout_prob1=0.25).to(device)
        elif model_type == 'resnet18':
            model = resnet18(pretrained=False, num_classes=10)

            # Need to change the input layer to accept a single greyscale channel
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model = model.to(device)
        else:
            raise NotImplementedError()

        model.load_state_dict(torch.load(path))
    else:
        model = torch.load(path)
        model = model.to(device)

    return model


def load_val_train_split(split_path):
    path = os.path.join(split_path, 'val_indices.pkl')
    with open(path, 'rb') as f:
        val_indices = pickle.load(f)

    print('============ Loaded validation indices from {}'.format(path))

    path = os.path.join(split_path, 'train_indices.pkl')
    with open(path, 'rb') as f:
        train_indices = pickle.load(f)

    print('============ Loaded train indices from {}'.format(path))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def main():
    from create_mask_voting import MaskedDataset, load_mask

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MNIST data to')

    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout probability to use (default: 0.25)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', type=str, default=None, help='Path to save model to (default: None)')
    parser.add_argument('--plot', type=str, default=None, help='Name of Visdom plot (default: None)')

    parser.add_argument('--model', choices=['large-cnn', 'small-cnn', 'mlp', 'gabor', 'resnet18'], default='large-cnn',
                        help='Model to train (default: large-cnn)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset for training')
    parser.add_argument('--use-swa', action='store_true', help='Use Stochastic Weight Averaging optimiser')
    parser.add_argument('--default-init', action='store_true', help='Initialise model with default weights')
    parser.add_argument('--val-split-path', type=str, default=None, help='Path to train/val indices split (if using)')

    parser.add_argument('--mask', type=str, default=None, help='Path to mask dataset with during training')
    parser.add_argument('--mask-per-class', type=str, default=None, help='Basepath to masks for each dataset class')

    parser.add_argument('--param-save-path', type=str, default=None,
                        help='Path to save model parameters during training to')

    parser.add_argument('--output-save-path', type=str, default=None, help='Path to save outputs during training to')

    parser.add_argument('--epoch-save-path', type=str, default=None, help='Path to save model after every epoch to')

    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.mask is not None and args.mask_per_class is not None:
        raise ValueError('Only one of --mask and --mask-per-class may be used at once')

    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot)

    if args.output_save_path is not None and args.val_split_path is None:
        raise Exception('If saving outputs, a validation set must be provided')

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_set = datasets.MNIST(args.data_path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

    if args.mask is not None:
        mask = load_mask(args.mask)
        train_set = MaskedDataset(train_set, mask)
        print('Using mask from {}'.format(args.mask))

    if args.mask_per_class is not None:
        # Get all files in the directory
        paths = [os.path.join(args.mask_per_class, f) for f in os.listdir(args.mask_per_class) if f[-4:] == '.pkl']
        masks = {}

        for path in paths:
            c = path[-5:-4]
            masks[c] = load_mask(path)

        train_set = MaskedDataset(train_set, masks, transforms=transforms.Compose([
                                   transforms.Normalize((0.2329,), (0.9119,))
                               ]))

    if args.val_split_path is not None:
        train_sampler, val_sampler = load_val_train_split(args.val_split_path)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                                   sampler=train_sampler, **kwargs)
        val_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                                 sampler=val_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                                   **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=args.shuffle, **kwargs)

    if args.default_init:
        torch.manual_seed(1)

    loss_fn = nn.NLLLoss()
    resnet = False
    if args.model == 'large-cnn':
        model = Net(dropout_prob1=args.dropout, dropout_prob2=args.dropout).to(device)
    elif args.model == 'small-cnn':
        model = SmallNet(dropout_prob1=args.dropout).to(device)
    elif args.model == 'mlp':
        model = MLP(dropout_prob=args.dropout).to(device)
    elif args.model == 'gabor':
        model = GaborNet(dropout_prob1=args.dropout).to(device)
    elif args.model == 'resnet18':
        resnet = True
        model = resnet18(pretrained=False, num_classes=10)

        # Need to change the input layer to accept a single greyscale channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        model = model.to(device)

        # Need to use Cross Entropy loss as we have logits as the output
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    num_params = sum(p.numel() for p in model.parameters())
    print('Model arch {} has {} parameters'.format(args.model, num_params))
    torch.manual_seed(args.seed)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.use_swa:
        print('=========== Using SWA')
        swa = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)
    else:
        swa = None

    param_path = args.param_save_path
    if param_path is not None:
        param_names = list(model.state_dict().keys())

        columns = {}
        for p in param_names:
            num_params = len(list(model.state_dict()[p].flatten()))
            current_columns = ['epoch', 'batch_num'] + [str(i) for i in range(num_params)]
            columns[p] = current_columns

        param_names_save_path = os.path.join(param_path, 'param_names.pkl')

        with open(param_names_save_path, 'wb') as f:
            pickle.dump(param_names, f)

        print('=============== Param names saved to {}'.format(param_names_save_path))

        save_path = os.path.join(param_path, 'columns.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(columns, f)

        print('=============== Columns dict saved to {}'.format(save_path))

    epoch_save_path = args.epoch_save_path
    if epoch_save_path is not None:
        dir_name = 'model{}_seed{}_dropout{}_shuffle{}'.format(args.model, args.seed, args.dropout, args.shuffle)
        epoch_save_path = os.path.join(epoch_save_path, dir_name)
        os.makedirs(epoch_save_path, exist_ok=True)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, loss_fn, epoch, verbose=args.verbose, swa=swa,
                           param_path=args.param_save_path)
        test_loss, acc = test(model, device, test_loader, loss_fn, resnet=resnet, verbose=args.verbose)
        scheduler.step()

        if args.plot is not None:
            plotter.plot('loss', 'train', 'loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'loss', epoch, test_loss)
            plotter.plot('accuracy', 'test', 'acc', epoch, acc)

        if args.output_save_path is not None:
            all_outputs = []

            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)

                    output = model(data)
                    all_outputs.append(output.cpu())

                all_outputs = torch.cat(all_outputs)

            path = os.path.join(args.output_save_path, 'output-epoch{}.pkl'.format(epoch))
            with open(path, 'wb') as f:
                pickle.dump(all_outputs, f)

            if args.verbose:
                print('============ Saved outputs for epoch {} to {}'.format(epoch, path))

        if epoch_save_path is not None:
            this_epoch_save_path = os.path.join(epoch_save_path, '{}.pkl'.format(epoch))
            torch.save(model.state_dict(), this_epoch_save_path)
            print('============ Save model at end of epoch {} to {}'.format(epoch, this_epoch_save_path))

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()