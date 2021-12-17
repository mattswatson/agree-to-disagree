import argparse
import pickle
import os

from utils import calculate_all_shap_importance
from XrayDataset import XrayDataset, XrayDatasetHDF5

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description='Calculate and save SHAP values for a model on a given dataset')
    parser.add_argument('model', type=str, help='Path to PyTorch model to interpret')
    parser.add_argument('save_path', type=str, help='Path to save calculated SHAP values to')
    parser.add_argument('--data-path', type=str, default='../../data', help='Path to download CXR data to')
    parser.add_argument('--no-flatten', action='store_true', help='Don\'t flatten the SHAP values')

    parser.add_argument('--baseline', choices=['zero', 'random', 'mean'], default='random',
                        help='Baseline to use for SHAP')

    parser.add_argument('--label', help='Label to classify', type=str, default='Edema')
    parser.add_argument('--checkpoint', action='store_true', help='If loading checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flatten = not args.no_flatten

    print('============= Using device', device)

    print('============= Loading model from', args.model)
    model = models.densenet121(pretrained=True)

    # Set Densenet to have the correct number of output classes
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    if args.checkpoint:
        checkpoint = torch.load(args.model)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(args.model))
    model = model.to(device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformList = []
    transformList.append(transforms.Resize(256))
    transformList.append(transforms.RandomResizedCrop(224))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)

    dataset = XrayDataset(args.data_path, split='validate', label=args.label, ignore_empty_labels=False,
                          transform=transformSequence)

    labels = [0, 1]

    if args.baseline == 'mean':
        sample = dataset[0][0]
        baseline = torch.zeros_like(sample[None, :, :, :])
        baseline[:, 0, :, :] = 0.485
        baseline[:, 1, :, :] = 0.456
        baseline[:, 2, :, :] = 0.406
    else:
        baseline = args.baseline

    for label in labels:
        shap_label_samples = calculate_all_shap_importance(model, dataset, device, multiclass=True,
                                                           flatten=flatten, target=label, baseline=baseline)

        path = os.path.join(args.save_path, '{}.pkl'.format(label))
        with open(path, 'wb') as f:
            print('============= Saving all SHAP values to', path)
            pickle.dump(shap_label_samples, f)


if __name__ == '__main__':
    main()