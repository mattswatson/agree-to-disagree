from ot import wasserstein_1d
import statsmodels.api as smtools
from divergence import *

import argparse
import os
import torch
import numpy
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_model_info(path):
    current_model = path.split('/')[-1]
    model_info = current_model.split('_')
    model_info = {'seed': model_info[1][4:], 'dropout': model_info[2][7:], 'shuffle': model_info[3][7:]}

    return model_info


def main():
    parser = argparse.ArgumentParser(description='Calculate WD divergence for all model pairs')
    parser.add_argument('model_path', type=str, help='Root directory containing all model sub-directories')
    parser.add_argument('fig_path', type=str, help='Directory to save figures to')
    parser.add_argument('--ignore-multiple-diff', action='store_true',
                        help='Ignore pairs where multiple changes have been made')
    parser.add_argument('--metric', choices=['wd', 'jsd'], default='wd', help='Metric to use')
    args = parser.parse_args()

    model_types = [f.path for f in os.scandir(args.model_path) if f.is_dir()]

    for p in model_types:
        # Get a list of all the models of this type
        models = [f.path for f in os.scandir(p) if f.is_dir()]

        # Get unique pairs of these models (no need to repeat pairs)
        pairs = [[models[i], models[j]] for i in range(len(models)) for j in range(i + 1, len(models))]

        # Load an example model just so we know what layers to look for
        model = torch.load(os.path.join(pairs[0][0], '1.pkl'))
        layer_names = model.keys()

        plots = {name: {} for name in layer_names}

        for pair in pairs:
            wd = {name: [] for name in layer_names}
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

            for epoch in range(1, num_epochs + 1):
                model1 = torch.load(os.path.join(pair[0], '{}.pkl'.format(epoch)))
                model2 = torch.load(os.path.join(pair[1], '{}.pkl'.format(epoch)))

                for layer in layer_names:
                    param1 = model1[layer].cpu().numpy().flatten()
                    param2 = model2[layer].cpu().numpy().flatten()

                    if args.metric == 'wd':
                        wd_curr = wasserstein_1d(param1, param2)
                    elif args.metric == 'jsd':
                        wd_curr = jensen_shannon_divergence_from_samples(param1, param2, discrete=False)
                    else:
                        raise NotImplementedError('Metric {} not implemented'.format(args.metric))

                    print('Metric for layer {} is {}'.format(layer, wd_curr))
                    wd[layer].append(wd_curr)

            # We've now got all of our distances, add them to the correct variables for plotting
            for layer in layer_names:
                plots[layer][line_name] = wd[layer]

        for plot in plots:
            plt.clf()
            rows = []

            for pair in plots[plot]:
                wd = plots[plot][pair]

                for i in range(len(wd)):
                    row = [i + 1, wd[i], pair]
                    rows.append(row)

            if args.metric == 'wd':
                metric_title = 'Wasserstein Ditance'
                metric_axis = 'WD'
            else:
                metric_title = 'Jensen-Shannon Divergence'
                metric_axis = 'JSD'

            df = pd.DataFrame(data=rows, columns=['epoch', metric_axis, 'pair'])

            sns.set(rc={'figure.figsize': (15.7, 11.27)})

            g = sns.lineplot(data=df, x='epoch', y=metric_axis, hue='pair')
            g.legend(loc='upper right')
            g.set_title('{} between variations on layer {} of {}'.format(metric_title, plot, p.split('/')[-1]))

            save_to = os.path.join(args.fig_path, p.split('/')[-1])
            os.makedirs(save_to, exist_ok=True)
            save_to = os.path.join(save_to, '{}.png'.format(plot))

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(save_to, bbox_inches='tight')

            print('============== Saved plot to {}'.format(save_to))


if __name__ == '__main__':
    main()
