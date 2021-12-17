import argparse
import pickle
import os

from utils import calculate_all_ig_importance
from mnist import Net, SmallNet, MLP, load_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def main():
    parser = argparse.ArgumentParser(description='Calculate and save IG values for a model on a given dataset')
    parser.add_argument('model', type=str, help='Path to PyTorch model to interpret')
    parser.add_argument('save_path', type=str, help='Path to save calculated IG values to')
    parser.add_argument('--data-path', type=str, default='../../data', help='Path to download MNIST data to')
    parser.add_argument('--no-flatten', action='store_true', help='Don\'t flatten the IG values')

    parser.add_argument('--state-dict', action='store_true', help='If loading state dict instead of whole model object')
    parser.add_argument('--model-class', choices=['large-cnn', 'small-cnn', 'mlp'], default='large-cnn',
                        help='Model to train, only use if passing state dict (default: large-cnn)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flatten = not args.no_flatten

    print('============= Using device', device)

    print('============= Loading model from', args.model)
    model = load_model(args.model, args.model_class, args.state_dict, device)
    model = model.to(device)

    dataset = datasets.MNIST(args.data_path, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    labels = torch.unique(dataset.targets)

    for label in labels:
        label = torch.tensor(label).to(device)
        ig_label_samples = calculate_all_ig_importance(model, dataset, device, multiclass=True, flatten=flatten,
                                                       target=label)

        path = os.path.join(args.save_path, '{}.pkl'.format(label))
        with open(path, 'wb') as f:
            print('============= Saving all IG values to', path)
            pickle.dump(ig_label_samples, f)


if __name__ == '__main__':
    main()
