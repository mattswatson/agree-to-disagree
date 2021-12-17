import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models as tv_models
from torch.utils.data import DataLoader

import argparse
import os
import operator
import numpy
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cca_core import *
from pwcca import *

from XrayDataset import XrayDataset
from finetune_densenet import get_cxr_datasets


def get_model_info(path):
    seed = path.split('/')[-1][-1]
    model_info = {'seed': seed}
    return model_info


def main():
    parser = argparse.ArgumentParser(description='Calculate CCA for a pair of models')
    parser.add_argument('model_path', type=str, help='Root directory containing all model sub-directories')
    parser.add_argument('fig_path', type=str, help='Directory to save figures to')
    parser.add_argument('--data-path', type=str, default='/media/hdd/mimic-cxr-jpg', help='Path to MIMIC-CXR data')
    parser.add_argument('--metric', choices=['cca', 'pwcca'], default='cca')

    parser.add_argument('--label', help='Label to classify', type=str, default='Edema')
    parser.add_argument('--ignore-empty-labels', help='Ignore samples where the diagnosis is not mentioned',
                        action='store_true')
    parser.add_argument('--interval', type=int, default=1, help='Look at layers every x epochs')
    args = parser.parse_args()

    device = torch.device("cuda")

    layers = ['features.conv0', 'features.denseblock1', 'features.denseblock2', 'features.denseblock3',
              'features.denseblock4', 'classifier']

    test_dataset, _, _ = get_cxr_datasets(args.data_path, args.label, None, args.ignore_empty_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4, num_workers=1, pin_memory=True)

    # Get a list of all the models
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

        for epoch in range(1, num_epochs, args.interval):
            if epoch > 49:
                break

            # Set up basic models
            model1 = tv_models.densenet121(pretrained=True)

            # Set Densenet to have the correct number of output classes
            num_features = model1.classifier.in_features
            model1.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

            checkpoint = torch.load(os.path.join(pair[0],
                                                 'checkpoint{}-labelEdema-epochs50-lr0.0005-batchsize16.tar'.format(
                                                     epoch)))

            model1.load_state_dict(checkpoint['model_state_dict'])
            model1.eval()
            model1 = model1.to(device)

            model2 = tv_models.densenet121(pretrained=True)

            # Set Densenet to have the correct number of output classes
            num_features = model2.classifier.in_features
            model2.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

            checkpoint = torch.load(os.path.join(pair[1],
                                                 'checkpoint{}-labelEdema-epochs50-lr0.0005-batchsize16.tar'.format(
                                                     epoch)))

            model2.load_state_dict(checkpoint['model_state_dict'])
            model2.eval()
            model2 = model2.to(device)

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
                operator.attrgetter(l)(model1).register_forward_hook(get_activation('model1-{}'.format(l)))
                operator.attrgetter(l)(model2).register_forward_hook(get_activation('model2-{}'.format(l)))

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
        g.set_title('{} between variations on layer {} of {}'.format(metric_title, plot,
                                                                     args.model_path.split('/')[-1]))

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