from utils import VisdomLinePlotter, AverageMeter, calculate_all_shap_importance, calculate_all_ig_importance
from finetune_densenet import train, get_cxr_datasets
from XrayDataset import XrayDataset

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.optim.lr_scheduler import StepLR
import os
import pickle
import numpy as np


def test(models, device, test_loader, verbose=False, weights=None):
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Make sure we have the correct data type
            data = data.type(torch.FloatTensor)

            data, target = data.to(device), target.to(device)
            print('target:', target)

            outputs = torch.stack([model(data) for model in models])
            print('outputs:', outputs)

            # Sum across the ensemble models
            if weights is None:
                summed_outputs = torch.sum(outputs, dim=0)
            else:
                summed_outputs = torch.tensordot(outputs, weights, dims=([0], [0]))

            print('summed outputs:', summed_outputs)

            # Then get the max
            _, pred = torch.max(summed_outputs, 1)
            print('pred:', pred)

            correct = torch.sum(pred == target.view_as(pred))

            num_samples += len(data)
            print('--------------------')

    if verbose:
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct,num_samples, 100. * correct / num_samples))

    acc = 100. * correct / num_samples
    return acc


def ensemble_expl(models, paths, ensemble_path, dataset, device, flatten, labels, baseline='random', method='shap'):
    if baseline == 'mean':
        sample = dataset[0][0]
        baseline = torch.zeros_like(sample[None, :, :, :])
        baseline[:, 0, :, :] = 0.485
        baseline[:, 1, :, :] = 0.456
        baseline[:, 2, :, :] = 0.406

    for label in labels:
        for i in range(len(models)):
            model = models[i]
            path = paths[i]

            if method == 'shap':
                expl_vals = calculate_all_shap_importance(model, dataset, device, multiclass=True,  flatten=flatten,
                                                          target=label, baseline=baseline)
            elif method == 'ig':
                expl_vals = calculate_all_ig_importance(model, dataset, device, multiclass=True, flatten=flatten,
                                                        target=label)
            else:
                raise Exception('Ensemble explanations only implement SHAP or IG')

            path = os.path.join(path, '{}.pkl'.format(label))
            with open(path, 'wb') as f:
                print('============= Saving all {} values to {}'.format(method.upper(), path))
                pickle.dump(expl_vals, f)

    print('============= Calculating {} value of ensembled model'.format(method.upper()))

    # Currently we are just taking an equal average, so we just do the same to the explanation values
    for label in labels:
        summed = []
        for path in paths:
            filepath = os.path.join(path, '{}.pkl'.format(label))
            with open(filepath, 'rb') as f:
                expl_vals = pickle.load(f)

            # If this is our first model, create an empty array of the same shape so we can easily add
            if len(summed) == 0:
                summed = np.zeros_like(expl_vals)

            summed += expl_vals

        ensemble_expl_vals = np.divide(summed, len(models))
        ensemble_filepath = os.path.join(ensemble_path, '{}.pkl'.format(label))
        with open(ensemble_filepath, 'wb') as f:
            print('============= Saving all ensemble {} values to {}'.format(method.upper(), ensemble_filepath))
            pickle.dump(ensemble_expl_vals, f)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MIMIC-CXR model average ensemble')

    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to get MIMIC-CXR data from')
    parser.add_argument('--plot', type=str, default=None, help='Name of Visdom plot (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset for training')
    parser.add_argument('--models-dir', type=str, default='./models', help='Path to directory containing models')
    parser.add_argument('--weighted', action='store_true', help='Calculate and use weights for the ensemble')

    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parser.add_argument('--label', help='Label to classify', type=str, default='Edema')
    parser.add_argument('--ignore-empty-labels', help='Ignore samples where the diagnosis is not mentioned',
                        action='store_true')

    parser.add_argument('--shap', action='store_true', help='Also calculate SHAP for the ensemble models')
    parser.add_argument('--shap-dir', type=str, default='./ensemble-shap', help='Directory to save SHAP values to')
    parser.add_argument('--no-flatten', action='store_true', help='Don\'t flatten the SHAP values')
    parser.add_argument('--baseline', choices=['zero', 'random', 'mean'], default='random',
                        help='Baseline to use for SHAP')
    parser.add_argument('--method', choices=['shap', 'ig'], default='shap',
                        help='Explanation method to use (default: shap)')
    parser.add_argument('--val-split-path', type=str, default=None, help='Path to train/val indices split (if using)')

    parser.add_argument('--checkpoints', action='store_true', help='Use if loading models from saved checkpoints')

    parser.add_argument('--split', choices=['val', 'test'], default='val', help='Dataset split to use (default: val)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.plot is not None:
        plotter = VisdomLinePlotter(args.plot)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_dataset, _, _ = get_cxr_datasets(args.data_path, args.label, None, args.ignore_empty_labels)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # Load the models
    models = []
    shap_paths = []
    for file in os.listdir(args.models_dir):
        if args.checkpoints:
            extension = '.tar'
        else:
            extension = '.pth'

        if file.endswith(extension):
            path = os.path.join(args.models_dir, file)
            model = torchvision_models.densenet121(pretrained=True)

            # Set Densenet to have the correct number of output classes
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

            if args.checkpoints:
                checkpoint = torch.load(path)
                state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(torch.load(args.model))

            model = model.to(device)
            model.eval()
            models.append(model)
            test(models, test_loader, verbose=True)

            # Also generate the SHAP paths if we're calculating the SHAP values
            if args.shap:
                shap_path = os.path.join(args.shap_dir, file[:-4])
                os.makedirs(shap_path, exist_ok=True)
                shap_paths.append(shap_path)

    # We only need to go through the test data, as there is no training to do
    #acc = test(models, device, test_loader, verbose=args.verbose)
    #print('Accuracy of ensemble model across whole test set is {}'.format(acc))

    if args.shap:
        if args.split == 'val':
            _, _, dataset = get_cxr_datasets(args.data_path, args.label, None, args.ignore_empty_labels)
        else:
            dataset, _, _ = get_cxr_datasets(args.data_path, args.label, None, args.ignore_empty_labels)

        flatten = not args.no_flatten

        print('================ Calculating all explanation values for the ensembled models')
        ensemble_expl(models, shap_paths, args.shap_dir, dataset, device, flatten, [0, 1],
                      baseline=args.baseline, method=args.method)


if __name__ == '__main__':
    main()
