import torch
import torch.nn as nn
from torchvision import datasets, transforms

import argparse
import os
import numpy
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cca_core import *
from pwcca import *

from model_wd import get_model_info

from mnist import load_model


def main():
    parser = argparse.ArgumentParser(description='Calculate CCA for a pair of models')
    parser.add_argument('model_path', type=str, help='Root directory containing all model sub-directories')
    parser.add_argument('fig_path', type=str, help='Directory to save figures to')
    parser.add_argument('--model-type', choices=['large-cnn', 'small-cnn', 'mlp'], default='large-cnn',
                        help='Model to train (default: large-cnn)')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MNIST data to')
    parser.add_argument('--metric', choices=['cca', 'pwcca'], default='cca')
    parser.add_argument('--interval', type=int, default=1, help='Look at layers every x epochs')
    args = parser.parse_args()

    device = torch.device("cuda")

    # We could possibly do this programmatically, but it would be difficult to ignore functional layers
    if args.model_type == 'large-cnn':
        layers = ['conv1', 'conv2', 'fc1', 'fc2']
    elif args.model_type == 'small-cnn':
        layers = ['conv1', 'fc1']
    elif args.model_type == 'mlp':
        layers = ['fc1', 'fc2', 'fc3']
    else:
        raise NotImplementedError('Model type {} not supported!'.format(args.model_type))

    kwargs = {'num_workers': 1, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=32, shuffle=False, **kwargs)

    # Get a list of all the models of this type
    models = [f.path for f in os.scandir(args.model_path) if f.is_dir()]

    # Get unique pairs of these models (no need to repeat pairs)
    pairs = [[models[i], models[j]] for i in range(len(models)) for j in range(i + 1, len(models))]

    plots = {name: {} for name in layers}
    for pair in pairs:
        wd = {name: [] for name in layers}
        model_names = [i.split('/')[-1] for i in pair]
        model_infos = [get_model_info(model_names[0]), get_model_info(model_names[1])]

        # Find which parameter was changed during training
        diffs = [name for name in model_infos[0] if model_infos[0][name] != model_infos[1][name]]

        # Don't bother if more than 1 change has been made during training
        if len(diffs) > 1:
            continue

        line_name = ''
        for diff in diffs:
            if line_name != '':
                line_name += '-'

            line_name += '{}{}/{}{}'.format(diff, model_infos[0][diff], diff, model_infos[1][diff])

        # Get number of epochs
        num_epochs = len([f.path for f in os.scandir(pair[0]) if f.is_file()])

        for epoch in range(1, num_epochs + 1, args.interval):
            model1 = load_model(os.path.join(pair[0], '{}.pkl'.format(epoch)), args.model_type, state_dict=True,
                                device=device)
            model1.eval()

            model2 = load_model(os.path.join(pair[1], '{}.pkl'.format(epoch)), args.model_type, state_dict=True,
                                device=device)
            model2.eval()

            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    # If we are looking at a convolutional layer, we want to flatten it so we can use CCA
                    if len(output.shape) > 2:
                        output = torch.reshape(output, (output.shape[0] * output.shape[2] * output.shape[3], output.shape[1]))

                    if name in activation:
                        activation[name] = torch.cat((activation[name], output.detach()))
                    else:
                        activation[name] = output.detach()

                return hook

            for l in layers:
                print('============== Inspecting layer {}'.format(l))
                getattr(model1, l).register_forward_hook(get_activation('model1-{}'.format(l)))
                getattr(model2, l).register_forward_hook(get_activation('model2-{}'.format(l)))

            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    model1(data)
                    model2(data)

            for key in activation:
                if torch.isnan(activation[key]).any() or torch.isinf(activation[key]).any():
                    raise Exception('{} activation contains nan/inf'.format(key))

                activation[key] = activation[key].cpu().numpy()

                # Shape should be neurons x datapoints
                activation[key] = np.swapaxes(activation[key], 0, 1)

            for l in layers:
                print('activation shape: {}'.format(activation['model1-{}'.format(l)].shape))

                if args.metric == 'cca':
                    cca = get_cca_similarity(activation['model1-{}'.format(l)], activation['model2-{}'.format(l)],
                                                  epsilon=1e-7)
                    distance = np.mean(cca['cca_coef1'])
                elif args.metric == 'pwcca':
                    mean_pwcca, w, _ = compute_pwcca(activation['model1-{}'.format(l)], activation['model2-{}'.format(l)],
                                                  epsilon=1e-7)
                    distance = mean_pwcca
                else:
                    raise NotImplementedError('Method {} not implemented!'.format(args.metric))

                print('CCA distance in layer {}: {}'.format(l, distance))

                wd[l].append(distance)

        for layer in layers:
            plots[layer][line_name] = wd[layer]

    for plot in plots:
        plt.clf()
        rows = []

        for pair in plots[plot]:
            wd = plots[plot][pair]

            for i in range(len(wd)):
                row = [i + 1, wd[i], pair]
                rows.append(row)

        if args.metric == 'cca':
            metric_title = 'CCA Similarity'
            metric_axis = 'CCA Similarity'
        else:
            metric_title = 'PWCCA Similarity'
            metric_axis = 'PWCCA'

        df = pd.DataFrame(data=rows, columns=['epoch', metric_axis, 'pair'])

        sns.set(rc={'figure.figsize': (15.7, 11.27)})

        g = sns.lineplot(data=df, x='epoch', y=metric_axis, hue='pair')
        g.legend(loc='upper right')
        g.set_title('{} between variations on layer {} of CNN on MNIST'.format(metric_title, plot))

        save_to = os.path.join(args.fig_path, args.model_path.split('/')[-1])
        os.makedirs(save_to, exist_ok=True)
        save_to = os.path.join(save_to, '{}.png'.format(plot))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(save_to, bbox_inches='tight')

        print('============== Saved plot to {}'.format(save_to))

        # Also save raw data
        save_to = os.path.join(args.fig_path, args.model_path.split('/')[-1], '{}.csv'.format(plot))
        df.to_csv(save_to)


if __name__ == '__main__':
    main()