from visdom import Visdom
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from captum.attr import GradientShap, DeepLift, IntegratedGradients
from shap import KernelExplainer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Visdom Plotting
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.vis = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def plot_matplotlib(self, plot_name, plt):
        self.plots[plot_name] = self.vis.matplot(plt,env=self.env)

    def plot_text(self, text, title='Text'):
        self.vis.text(text, env=self.env, opts=dict(title=title))


def flatten_item(item):
    return item.flatten()

def calculate_all_shap_importance(model, dataset, device, baseline='random', multiclass=False, flatten=False,
                                  target=None, return_inputs=False):
    model = model.eval()

    shap = GradientShap(model)

    # Set up the dataloader for our dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get the shape of our dataset so we can make our baseline if needed
    shape = dataset[0][0].size()

    # If we're not given a baseline, just use the all zero tensor
    if baseline is None:
        baseline = torch.zeros((1, shape[0]))
        baseline = baseline.to(device)
    elif type(baseline) == list or type(baseline) == torch.Tensor:
        baseline = torch.tensor(baseline)
        baseline = baseline.to(device)
    elif baseline == 'random':
        # Generate a distribution of random data
        # Note that this only really works for images/continuous data
        sample, label = next(iter(dataloader))
        baseline = torch.cat([sample * 0, sample * 1])
        baseline = baseline.to(device)

    print("======================= Calculating SHAP Values (GradientShap Approximation)...")

    shap_all_samples = []
    all_inputs = []
    with tqdm(total=len(dataset)) as progress:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if return_inputs:
                all_inputs.append(inputs)
            
            if target != targets[0].cpu() and multiclass:
                continue

            if not multiclass:
                shap_vals = shap.attribute(inputs, baselines=baseline)
            else:
                shap_vals = shap.attribute(inputs, baselines=baseline, target=targets)

            if flatten:
                shap_vals = flatten_item(shap_vals[0])
                shap_vals = shap_vals.cpu().detach().numpy()
            else:
                shap_vals = shap_vals[0]
                shap_vals = shap_vals.cpu().detach().numpy()

            shap_all_samples.append(shap_vals)

            progress.update(1)

    print("======================= SHAP Values calculated!")

    if return_inputs:
        return shap_all_samples, all_inputs
    else:
        return shap_all_samples


def calculate_average_shap_importance(model, dataset, device, baseline=None, multiclass=False):
    shap_all_samples = calculate_all_shap_importance(model, dataset, device, baseline, multiclass)

    avg_shap_vals = np.mean(shap_all_samples, axis=0)
    avg_shap_vals_dict = {i: avg_shap_vals[i] for i in range(len(avg_shap_vals))}

    return avg_shap_vals_dict


def calculate_average_shap_importance_from_array(shap_all_samples, absolute=True):
    if absolute:
        avg_shap_vals = np.mean([np.absolute(sample) for sample in shap_all_samples], axis=0)
    else:
        avg_shap_vals = np.mean(shap_all_samples, axis=0)

    avg_shap_vals_dict = {i: avg_shap_vals[i] for i in range(len(avg_shap_vals))}

    return avg_shap_vals_dict


def plot_shap_vals_dict(shap_vals, feature_names=None, axis_title='Feature', title='SHAP values', xtick_nth=1,
                        ylim=None):
    if feature_names is None:
        feature_names = [str(i) if i % xtick_nth == 0 else str(0) for i in range(len(shap_vals))]

    x_pos = np.arange(len(feature_names))

    x_ticks = list(range(len(feature_names)))
    x_ticks = [x_tick if x_tick % xtick_nth == 0 else 0 for x_tick in x_ticks]

    shap_vals_arr = shap_vals.values()

    f, ax = plt.subplots()
    ax.bar(x_pos, shap_vals_arr, align='center')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(feature_names, rotation=90, horizontalalignment='center', fontsize='x-small')
    ax.set_xlabel(axis_title)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Only show every nth x label
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % xtick_nth != 0:
            label.set_visible(False)

    return ax


def calculate_all_deeplift_importance(model, dataset, device, baseline='random', multiclass=False, flatten=False,
                                      target=None, return_inputs=False, return_dict_array=False):
    model = model.eval()

    dl = DeepLift(model)

    # Set up the dataloader for our dataset
    dataloader = DataLoader(dataset, batch_size=1)

    # Get the shape of our dataset so we can make our baseline if needed
    shape = dataset[0][0].size()

    # If we're not given a baseline, just use the all zero tensor
    if baseline == 'zeros':
        sample, _ = next(iter(dataloader))
        baseline = torch.zeros(sample.shape)
    elif baseline == 'random':
        # Generate a distribution of random data
        # Note that this only really works for images/continuous data
        sample, label = next(iter(dataloader))
        baseline = torch.cat([sample * 0, sample * 1])
    else:
        raise NotImplementedError()

    baseline = baseline.to(device)

    print("======================= Calculating DeepLIFT Values...")

    dl_all_samples = []
    all_inputs = []
    with tqdm(total=len(dataset)) as progress:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if return_inputs:
                all_inputs.append(inputs)

            if target is not None and target != targets[0]:
                continue

            if not multiclass:
                dl_vals = dl.attribute(inputs, baselines=baseline)
            else:
                dl_vals = dl.attribute(inputs, baselines=baseline, target=targets)

            if flatten:
                dl_vals = flatten_item(dl_vals[0])
                dl_vals = dl_vals.cpu().detach().numpy()
            else:
                dl_vals = dl_vals[0]
                dl_vals = dl_vals.cpu().detach().numpy()

            if return_dict_array:
                dl_all_samples.append({'img': inputs[0], 'label': targets[0], 'expl': dl_vals})
            else:
                dl_all_samples.append(dl_vals)

            progress.update(1)

    print("======================= DeepLIFT Values calculated!")

    if return_inputs:
        return dl_all_samples, all_inputs
    else:
        return dl_all_samples


def calculate_average_deeplift_importance(model, dataset, device, baseline=None, multiclass=False):
    dl_all_samples = calculate_all_deeplift_importance(model, dataset, device, baseline, multiclass)

    avg_dl_vals = np.mean(dl_all_samples, axis=0)
    avg_dl_vals_dict = {i: avg_dl_vals[i] for i in range(len(avg_dl_vals))}

    return avg_dl_vals_dict


def calculate_average_deeplift_importance_from_array(dl_all_samples, absolute=True):
    if absolute:
        avg_dl_vals = np.mean([np.absolute(sample) for sample in dl_all_samples], axis=0)
    else:
        avg_dl_vals = np.mean(dl_all_samples, axis=0)

    avg_dl_vals_dict = {i: avg_dl_vals[i] for i in range(len(avg_dl_vals))}

    return avg_dl_vals_dict


def calculate_all_ig_importance(model, dataset, device, multiclass=False, flatten=False, target=None,
                                return_inputs=False):
    model = model.eval()

    ig = IntegratedGradients(model)

    # Set up the dataloader for our dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get the shape of our dataset so we can make our baseline
    shape = dataset[0][0].size()

    # For IG, we want our baseline to have a score of zero, which the all zero tensor satisfies
    baseline = torch.zeros((1, shape[0]))
    baseline = baseline.to(device)

    print("======================= Calculating Integrated Gradients Values...")

    ig_all_samples = []
    all_inputs = []
    with tqdm(total=len(dataloader)) as progress:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if return_inputs:
                all_inputs.append(inputs)

            if target is not None and target != targets[0]:
                continue

            if not multiclass:
                ig_vals = ig.attribute(inputs, baselines=baseline)
            else:
                ig_vals = ig.attribute(inputs, baselines=baseline, target=targets)

            if flatten:
                ig_vals = flatten_item(ig_vals[0])
                ig_vals = ig_vals.cpu().detach().numpy()
            else:
                ig_vals = ig_vals[0]
                ig_vals = ig_vals.cpu().detach().numpy()

            ig_all_samples.append(ig_vals)

            progress.update(1)

    print("======================= Integrated Gradients Values calculated!")

    if return_inputs:
        return ig_all_samples, all_inputs
    else:
        return ig_all_samples


def calculate_average_ig_importance(model, dataset, device, multiclass=False):
    ig_all_samples = calculate_all_ig_importance(model, dataset, device, multiclass)

    avg_ig_vals = np.mean(ig_all_samples, axis=0)
    avg_ig_vals_dict = {i: avg_ig_vals[i] for i in range(len(avg_ig_vals))}

    return avg_ig_vals_dict


def calculate_average_explanation_importance_from_array(ig_all_samples, absolute=True):
    if absolute:
        avg_ig_vals = np.mean([np.absolute(sample) for sample in ig_all_samples], axis=0)
    else:
        avg_ig_vals = np.mean(ig_all_samples, axis=0)

    avg_ig_vals_dict = {i: avg_ig_vals[i] for i in range(len(avg_ig_vals))}

    return avg_ig_vals_dict


def train_binary_svm(neg_class, pos_class, feature_names=None, test_split=0.3, kernel=None, save=None,
                     save_results=None, results_name='', scale=True):
    # Create a pandas dataframe for ease
    # In theory these should have the same length, but it's good to just check
    num_features = max([len(l) for l in neg_class] + [len(l) for l in pos_class])

    if feature_names is None:
        columns = [str(i) for i in range(num_features)]
    else:
        columns = feature_names

    df = pd.DataFrame(neg_class + pos_class, columns=columns)
    labels = pd.DataFrame.from_records([[0] for _ in range(len(neg_class))] +
                                       [[1] for _ in range(len(pos_class))])

    df = df.fillna(0)

    # Split the data up, normalise it
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)
    split = sss.split(df, labels)

    for train_indices, test_indices in split:
        data_train, labels_train = df.iloc[train_indices], labels.iloc[train_indices]
        data_test, labels_test = df.iloc[test_indices], labels.iloc[test_indices]

    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    # Train all SVMs, unless we're asked for a specific one
    possible_kernels = ['linear', 'poly', 'rbf', 'sigmoid'] if kernel is None else [kernel]

    for k in possible_kernels:
        print("======================= Training {} SVM".format(k))

        svm = SVC(kernel=k, verbose=True, shrinking=True, cache_size=500)
        svm.fit(data_train, labels_train)

        if save is not None:
            filename = k + '-svm.pkl'
            path = os.path.join(save, filename)

            print("======================= Saving {} SVM to {}".format(k, path))

            with open(path, 'wb') as f:
                pickle.dump(svm, f)

        print("======================= Evaluating SVM")

        preds = svm.predict(data_test)

        conf_matrix = confusion_matrix(labels_test, preds).tolist()
        class_report = classification_report(labels_test, preds)
        results = "{} \n\n {}".format(conf_matrix, class_report)

        print(results)

        if save_results is not None:
            filename = '{}-results{}.txt'.format(k, results_name)
            path = os.path.join(save_results, filename)

            print("======================= Saving results to {}".format(path))

            with open(path, 'w') as f:
                f.write(results)


def create_shap_dataframe(feature_names, shap_vals_neg, shap_vals_pos, test_split):
    # Create a pandas dataframe for ease
    # In theory these should have the same length, but it's good to just check
    num_features = max([len(l) for l in shap_vals_neg] + [len(l) for l in shap_vals_pos])
    print('num features:', num_features)
    if feature_names is None:
        columns = [str(i) for i in range(num_features)]
    else:
        columns = feature_names

    df = pd.DataFrame(shap_vals_neg + shap_vals_pos, columns=columns)
    labels = pd.DataFrame.from_records([[0] for _ in range(len(shap_vals_neg))] +
                                       [[1] for _ in range(len(shap_vals_pos))])
    df = df.fillna(0)

    # Split the data up, normalise it
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)
    split = sss.split(df, labels)

    for train_indices, test_indices in split:
        data_train, labels_train = df.iloc[train_indices], labels.iloc[train_indices]
        data_test, labels_test = df.iloc[test_indices], labels.iloc[test_indices]

    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)

    return data_test, data_train, labels_test, labels_train


def train_lr(shap_vals_neg, shap_vals_pos, feature_names=None, test_split=0.3, save=None, save_results=None,
             results_name='', cv=False, n_jobs=1, verbose=0, scale=True):

    # Split the SHAP values into test and train sets
    data_test, data_train, labels_test, labels_train = create_shap_dataframe(feature_names, shap_vals_neg,
                                                                             shap_vals_pos, test_split)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    if cv:
        lr = LogisticRegressionCV(max_iter=1000, n_jobs=n_jobs, verbose=verbose)
    else:
        lr = LogisticRegression(verbose=verbose, max_iter=10000, solver='saga')
    lr.fit(data_train, labels_train)

    if save is not None:
        filename = 'lr-{}.pkl'.format(results_name)
        path = os.path.join(save, filename)

        print("======================= Saving LR to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(lr, f)

    print("======================= Evaluating LR")

    preds = lr.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    # Include the parameters if we used CV
    if cv:
        results = "{} \n\n {} \n\n {}".format(conf_matrix, class_report, lr.get_params())
    else:
        results = "{} \n\n {}".format(conf_matrix, class_report)

    print(results)

    if save_results is not None:
        filename = 'results-{}.txt'.format(results_name)
        path = os.path.join(save_results, filename)

        print("======================= Saving results to {}".format(path))

        with open(path, 'w') as f:
            f.write(results)


def train_lda(shap_vals_neg, shap_vals_pos, feature_names=None, test_split=0.3, save=None, save_results=None,
             results_name='', n_jobs=1, verbose=0, scale=True):

    # Split the SHAP values into test and train sets
    data_test, data_train, labels_test, labels_train = create_shap_dataframe(feature_names, shap_vals_neg,
                                                                             shap_vals_pos, test_split)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(data_train, labels_train)

    if save is not None:
        filename = 'lr-{}.pkl'.format(results_name)
        path = os.path.join(save, filename)

        print("======================= Saving LR to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(lr, f)

    print("======================= Evaluating LR")

    preds = lda.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    results = "{} \n\n {}".format(conf_matrix, class_report)
    print(results)

    if save_results is not None:
        filename = 'results-{}.txt'.format(results_name)
        path = os.path.join(save_results, filename)

        print("======================= Saving results to {}".format(path))

        with open(path, 'w') as f:
            f.write(results)
