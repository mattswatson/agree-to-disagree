# Use a model (typically the same as the original model) to classify explanations into the original classes of the
# dataset. Based on the idea of accuracy from https://christophm.github.io/interpretable-ml-book/properties.html
# For now, only works for MNIST

import argparse
import numpy as np
from tqdm import tqdm
import os
import pickle

from ExplanationDatasets import ShapDataset, LimeDataset, LrpDataset, DeepliftDataset
from utils import VisdomLinePlotter
from finetune_densenet import train, test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision import models

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy of an explanation method')
    parser.add_argument('--method', choices=['shap', 'lime', 'lrp', 'deeplift'], default='shap',
                        help='Explanation method to use')
    parser.add_argument('--path', type=str, help='Path to the calculated explanations', required=True)
    parser.add_argument('--path-prefix', type=str, default='', help='Prefix for SHAP files')
    parser.add_argument('--path-postfix', type=str, default='', help='Postfix for SHAP files')

    parser.add_argument('--plot', type=str, default=None, help='Name of Visdom plot (default: None)')

    # CNN options
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str, default=None, help='Path to save model to (default: None)')
    parser.add_argument('--train-split', type=float, default=0.8, help='Size of the train split as a fraction of '
                                                                       'whole dataset (default: 0.8)')
    parser.add_argument('--verbose', action='store_true', help='Log verbose output to console')
    parser.add_argument('--transform', action='store_true', help='Transform before training/testing')

    args = parser.parse_args()

    if args.method == 'shap':
        print('============= Using SHAP explanations')

        dataset = ShapDataset(path=args.path, path_prefix=args.path_prefix, path_postfix=args.path_postfix,
                              calculate=False, model_path=None)
    elif args.method == 'lime':
        print('============= Using LIME explanations')
        dataset = LimeDataset(args.path)
    elif args.method == 'lrp':
        print('============= Using LRP explanations')
        dataset = LrpDataset(args.path)
    elif args.method == 'deeplift':
        print('============= Using DeepLIFT explanations')

        # Find mean and std. of this set of DL vals
        dataset_no_transform = DeepliftDataset(path=args.path, path_prefix=args.path_prefix, normalise=None)

        loader = torch.utils.data.DataLoader(dataset_no_transform, batch_size=10, shuffle=False)

        mean = 0.
        std = 0.
        nb_samples = 0.

        for data, labels in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        dataset = DeepliftDataset(path=args.path, path_prefix=args.path_prefix, normalise=([mean], [std]))
    else:
        raise NotImplementedError()

    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Split dataset into test and train sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.train_split * dataset_size))

    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]


    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

    model = models.densenet121(pretrained=True)

    # Set Densenet to have the correct number of output classes
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    model = model.to(device)


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = nn.BCELoss()

    with tqdm(total=args.epochs) as progress:
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = train(model, device, train_loader, criterion, optimizer, verbose=args.verbose)
            test_loss, acc = test(model, device, test_loader, criterion, verbose=args.verbose)
            scheduler.step()

            if args.plot is not None:
                plotter.plot('loss', 'train', 'loss', epoch, train_loss)
                plotter.plot('loss', 'test', 'loss', epoch, test_loss)
                plotter.plot('accuracy', 'test', 'acc', epoch, acc)

            progress.update(1)

    if args.save_model is not None:
        print('============= Saved model to', args.save_model)
        torch.save(model, args.save_model)


if __name__ == '__main__':
    main()