import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import copy
import argparse
import time

from tqdm import tqdm

from XrayDataset import XrayDataset, XrayDatasetHDF5

from utils import VisdomLinePlotter, AverageMeter


def train(model, device, train_loader, criterion, optimiser, verbose=False):
    losses = AverageMeter()
    accuracy = AverageMeter()
    data_times = AverageMeter()
    batch_times = AverageMeter()

    model.train()

    end = time.time()
    total_correct = 0
    num_samples = 0
    with tqdm(total=len(train_loader)) as progress:
        for data, target in train_loader:
            data_times.update(time.time() - end)
            data = data.to(device)
            target = target.to(device).long()

            output = model(data)
            print('len data:', data.shape)
            print('len target:', target.shape)
            print('len target:', len(target))
            print('len output:', output.shape)

            loss = criterion(output, target)
            _, pred = torch.max(output, 1)

            if verbose:
                print('target:', target)
                print('pred:', pred)
                print('output:', output)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            #_, target = torch.max(target, 1)
            correct = torch.sum(pred == target.view_as(pred))
            total_correct += correct.item()
            num_samples += len(target)
            #accuracy.update(correct.item(), len(target))

            if verbose:
                print('correct:', correct)
                print('acc:', correct.item()/len(target))

            losses.update(loss.item(), len(target))
            batch_times.update(time.time() - end)
            end = time.time()

            progress.update(1)

    if verbose:
        print('Epoch took an average of {} on data loading and {} overall on a batch'.format(data_times.avg,
                                                                                             batch_times.avg))

    return losses.avg, total_correct / num_samples


def test(model, device, test_loader, criterion, verbose=False):
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.eval()

    total_correct = 0
    num_samples = 0
    with tqdm(total=len(test_loader)) as progress:
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).long()

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                _, pred = torch.max(output, 1)

            #_, target = torch.max(target, 1)
            correct = torch.sum(pred == target.view_as(pred))

            losses.update(loss.item(), len(target))
            #accuracy.update(correct.item(), len(target))
            total_correct += correct.item()
            num_samples += len(target)

            if verbose:
                print('target:', target)
                print('pred:', pred)
                print('output:', output)

            progress.update(1)

    return losses.avg, total_correct / num_samples


def get_cxr_datasets(data_path, label, hdf5_path=None, ignore_empty_labels=False):
    if hdf5_path is None:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.Resize(256))
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence = transforms.Compose(transformList)

        train_dataset = XrayDataset(data_path, split='train', label=label,
                                    ignore_empty_labels=ignore_empty_labels, transform=transformSequence)
        test_dataset = XrayDataset(data_path, split='test', label=label,
                                   ignore_empty_labels=False, transform=transformSequence)
        val_dataset = XrayDataset(data_path, split='validate', label=label,
                                  ignore_empty_labels=False, transform=transformSequence)
    else:
        print('=========== Using HDF5 dataset')
        train_dataset = XrayDatasetHDF5(data_path, hdf5_path, split='train', label=label,
                                        ignore_empty_labels=ignore_empty_labels)
        test_dataset = XrayDatasetHDF5(data_path, hdf5_path, split='test', label=label,
                                       ignore_empty_labels=False)
        val_dataset = XrayDatasetHDF5(data_path, hdf5_path, split='validate', label=label,
                                      ignore_empty_labels=False)
    return test_dataset, train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Train Densenet121 to classify CXR images')

    parser.add_argument('data_path', help='Path to root of MIMIC-CXR-JPG data', type=str)
    parser.add_argument('--hdf5-path', help='Path to HDF5 file containing images (if using)', type=str, default=None)

    parser.add_argument('--epochs', '-e', help='Number of epochs to train on', type=int, default=300)
    parser.add_argument('--batch-size', '-b', help='Batch size to train with', type=int, default=32)
    parser.add_argument('--lr', help='Learning rate during training', type=float, default=0.001)

    parser.add_argument('--label', help='Label to classify', type=str, default='Pneumonia')
    parser.add_argument('--ignore-empty-labels', help='Ignore samples where the diagnosis is not mentioned',
                        action='store_true')
    parser.add_argument('--save', '-s', help='Directory to save model to', type=str, default='./')

    parser.add_argument("--plot", type=str, default=None, help="name of visdom plot")
    parser.add_argument("--plot_server", type=str, default='localhost', help="visdom server address")
    parser.add_argument("--plot_port", type=int, default=8097, help='visdom server port')

    parser.add_argument("--checkpoint", type=str, default=None, help='directory to store checkpoint each epoch')
    parser.add_argument("--load-checkpoint", type=str, default=None, help='path to saved checkpoint to resume from')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--cudnn-benchmark', action='store_true', help='Enable cudnn benchmarking')

    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--no-shuffle', action='store_true', help='Don\'t shuffle the dataset')
    parser.add_argument('--default-init', action='store_true',
                        help='Initialise model with default weights (use when seed needs to be the same, '
                             'but different dataset shuffles)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.plot is not None:
        global plotter
        plotter = VisdomLinePlotter(args.plot, server=args.plot_server, port=args.plot_port)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('=========== Using device {}'.format(device))

    if device == 'cuda:0' and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    test_dataset, train_dataset, val_dataset = get_cxr_datasets(args.data_path, args.label, args.hdf5_path,
                                                                args.ignore_empty_labels)

    shuffle = not args.no_shuffle
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=args.batch_size, num_workers=args.workers,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    model = models.densenet121(pretrained=True)

    if args.default_init:
        torch.manual_seed(1)

    # Set Densenet to have the correct number of output classes
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())
    model = model.to(device)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    optimiser = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    current_epoch = 0

    if args.load_checkpoint is not None:
        print("Loading checkpoint from {}".format(args.load_checkpoint))
        checkpoint = torch.load(args.load_checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']

    for epoch in range(current_epoch, args.epochs):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimiser, verbose=args.verbose)
        test_loss, acc = test(model, device, test_loader, criterion, verbose=args.verbose)

        print("Epoch {}/{}: [Train loss: {:.4f}] [Test loss: {:.4f}] [Test accuracy: {}]".format(epoch, args.epochs,
                                                                                                 train_loss, test_loss,
                                                                                                 acc))

        if acc > best_acc:
            best_acc = acc
            best_model_weights = copy.deepcopy(model.state_dict())

        if args.plot is not None:
            plotter.plot('loss', 'train', 'loss', epoch, train_loss)
            plotter.plot('loss', 'test', 'loss', epoch, test_loss)
            plotter.plot('accuracy', 'test', 'acc', epoch, acc)
            plotter.plot('accuracy', 'train', 'acc', epoch, train_acc)

        if args.checkpoint is not None:
            checkpoint_path = os.path.join(args.checkpoint, "checkpoint{}-label{}-epochs{}-lr{}-batchsize{}.tar"
                                           .format(epoch, args.label, args.epochs, args.lr, args.batch_size))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

    model.load_state_dict(best_model_weights)
    print("Training finished, achieved best accuracy of {}".format(best_acc))

    save_path = os.path.join(args.save, 'densenet121-label{}-epochs{}-lr{}-batchsize{}.pth'.format(args.label,
                                                                                                   args.epochs,
                                                                                                   args.lr,
                                                                                                   args.batch_size))
    print('================ Saving model to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
