from ot import wasserstein_1d
import statsmodels.api as smtools

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
    parser.add_argument('model_path', type=str, help='Root directory containing all models')
    parser.add_argument('fig_path', type=str, help='Directory to save figures to')
    parser.add_argument('--ignore-multiple-diff', action='store_true',
                        help='Ignore pairs where multiple changes have been made')
    args = parser.parse_args()

    seeds = [f.path for f in os.scandir(args.model_path) if f.is_dir()]

    # Get unique pairs of skills
    pairs = [[seeds[i], seeds[j]] for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    plots = {}

    for pair in pairs:
        # Get all models in each pair
        models = [[f.path for f in os.scandir(pair[0]) if f.is_file()],
                  [f.path for f in os.scandir(pair[1]) if f.is_file()]]

        # Load an example model just so we know what layers to look for
        checkpoint = torch.load(models[0][0])
        layer_names = checkpoint['model_state_dict'].keys()

        wd = {name: [] for name in layer_names}

        line_name = '{}/{}'.format(pair[0].split('/')[-2], pair[1].split('/')[-2])

        # Get number of epochs
        num_epochs = min(len(models[0]), len(models[1]))

        for epoch in range(1, num_epochs + 1):
            model1_checkpoint = torch.load(os.path.join(pair[0], '{}.pkl'.format(epoch)))
            model2_checkpoint = torch.load(os.path.join(pair[1], '{}.pkl'.format(epoch)))

            model1 = model1_checkpoint['model_state_dict']
            model2 = model2_checkpoint['model_state_dict']

            for layer in layer_names:
                param1 = model1[layer].cpu().numpy().flatten()
                param2 = model2[layer].cpu().numpy().flatten()

                wd_curr = wasserstein_1d(param1, param2)
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

            df = pd.DataFrame(data=rows, columns=['epoch', 'wd', 'pair'])

            sns.set(rc={'figure.figsize': (15.7, 11.27)})

            g = sns.lineplot(data=df, x='epoch', y='wd', hue='pair')
            g.legend(loc='upper right')
            g.set_title('Wasserstein Distance between variations on layer {} of {}'.format(plot, p.split('/')[-1]))

            save_to = os.path.join(args.fig_path, p.split('/')[-1])
            os.makedirs(save_to, exist_ok=True)
            save_to = os.path.join(save_to, '{}.png'.format(plot))

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(save_to, bbox_inches='tight')

            print('============== Saved plot to {}'.format(save_to))


if __name__ == '__main__':
    main()
