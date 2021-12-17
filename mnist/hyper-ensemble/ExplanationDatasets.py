# Set of Dataset classes for the various explanation methods we use

from torch.utils.data import Dataset
import torch

from torchvision import datasets, transforms

import os
import pickle
import pandas as pd
import math


class ShapDataset(Dataset):
    def __init__(self, path=None, calculate=False, path_prefix='', path_postfix='', flatten=False, model_path=None,
                 return_inputs=False, state_dict=False):
        super(ShapDataset, self).__init__()

        self.flatten = flatten
        self.return_input = return_inputs

        if calculate:
            raise Exception('Calculating SHAP values is not supported for hyperensemble models.')

        if not calculate:
            if path is None:
                raise Exception('If not calculating SHAP values, must pass a path')

            # For SHAP, we sometimes have data with labels and sometimes not. For now, assume we have data with labels
            # That means that path is a directory that contains 0.pkl, 1.pkl, etc

            rows = []
            for label in range(0, 10):
                current_path = os.path.join(path, "{}{}{}.pkl".format(path_prefix, label, path_postfix))

                with open(current_path, 'rb') as f:
                    shap_vals = pickle.load(f)

                    for shap in shap_vals:
                        row = [shap, label]
                        rows.append(row)

            self.df = pd.DataFrame(rows, columns=['shap', 'label'])
        else:
            if model_path is None:
                raise Exception('Must pass a model for SHAP calculations')

            dataset = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print('============= Using device', device)

            if state_dict:
                model = model_path
            else:
                print('============= Loading model from', model_path)
                model = torch.load(model_path)

            model = model.to(device)

            rows = []
            labels = torch.unique(dataset.targets)

            for label in labels:
                if self.return_input:
                    shap_label_samples, inputs = calculate_all_shap_importance(model, dataset, device, multiclass=True,
                                                                               flatten=flatten, target=label,
                                                                               return_inputs=self.return_input)
                else:
                    shap_label_samples = calculate_all_shap_importance(model, dataset, device, multiclass=True,
                                                                       flatten=flatten, target=label,
                                                                       return_inputs=self.return_input)

                for i in range(len(shap_label_samples)):
                    sample = shap_label_samples[i]

                    if self.return_input:
                        img = inputs[i][0]
                        row = [sample, label, img]
                    else:
                        row = [sample, label]

                    rows.append(row)

            if self.return_input:
                self.df = pd.DataFrame(rows, columns=['shap', 'label', 'img'])
            else:
                self.df = pd.DataFrame(rows, columns=['shap', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]
        shap = torch.tensor(rows['shap'])
        label = torch.tensor(int(rows['label']))

        if not self.flatten:
            # If we want a 2D matrix, but are given a 1D matrix, make it 2D
            if len(shap.shape) == 1:
                # Assume that we can make a square image
                size = int(math.sqrt(shap.shape[0]))
                shap = shap.reshape(size, size)
                shap = shap[None, :, :]

        if self.return_input:
            img = rows['img']
            return shap, label, img
        else:
            return shap, label

class LimeDataset(Dataset):
    def __init__(self, path, flatten=False):
        super(LimeDataset, self).__init__()

        self.flatten = flatten

        with open(path, 'rb') as f:
            lime = pickle.load(f)

        rows = []
        for sample in lime:
            row = [sample['mask'], sample['label']]
            rows.append(row)

        self.df = pd.DataFrame(rows, columns=['mask', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]
        mask = torch.tensor(rows['mask'])
        mask = mask[None, :, :]
        label = torch.tensor(int(rows['label']))

        if self.flatten:
            mask = mask.flatten()

        return mask, label


class LrpDataset(Dataset):
    def __init__(self, path):
        super(LrpDataset, self).__init__()

        with open(path, 'rb') as f:
            lrp = pickle.load(f)

        rows = []
        for sample in lrp:
            all_expl = sample['expl']
            label = sample['label']

            # Get just the explanation for the correct class
            expl = all_expl[label][0][0]

            row = [expl, label]
            rows.append(row)

        self.df = pd.DataFrame(rows, columns=['expl', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]
        expl = torch.tensor(rows['expl'])
        expl = expl[None, :, :]
        label = torch.tensor(int(rows['label']))

        return expl, label


class DeepliftDataset(Dataset):
    def __init__(self, path, path_prefix=None, flatten=False, normalise=None, return_inputs=False):
        super(DeepliftDataset, self).__init__()

        self.flatten = flatten
        self.normalise = normalise
        self.return_inputs = return_inputs

        # If we have been given a prefix, we must have multiple files (one for each class)
        if path_prefix is not None:
            rows = []
            for label in range(0, 10):
                current_path = os.path.join(path, "{}{}.pkl".format(path_prefix, label))

                with open(current_path, 'rb') as f:
                    dl_vals = pickle.load(f)

                    for dl in dl_vals:
                        row = [dl, label]
                        rows.append(row)
        else:
            # Otherwise, we have a single file that is a list of dicts
            with open(path, 'rb') as f:
                all_dl_dicts = pickle.load(f)

                rows = []
                for d in all_dl_dicts:
                    if self.return_inputs:
                        row = [d['expl'], d['label'], d['img']]
                    else:
                        row = [d['expl'], d['label']]

                    rows.append(row)

        if self.return_inputs:
            self.df = pd.DataFrame(rows, columns=['expl', 'label', 'img'])
        else:
            self.df = pd.DataFrame(rows, columns=['expl', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]
        dl = torch.tensor(rows['expl'])
        label = torch.tensor(int(rows['label']))

        if not self.flatten:
            # If we want a 2D matrix, but are given a 1D matrix, make it 2D
            if len(dl.shape) == 1:
                # Assume that we can make a square image
                size = int(math.sqrt(dl.shape[0]))
                dl = dl.reshape(size, size)
                dl = dl[None, :, :]
        else:
            dl = dl.flatten()

        if self.normalise is not None:
            t = transforms.Compose([transforms.Normalize(*self.normalise)])
            dl = t(dl)

        if self.return_inputs:
            img = torch.tensor(rows['img'])
            return dl, label, img
        else:
            return dl, label


def get_subdirs(path):
    subdirs = []
    for f in os.scandir(path):
        if f.is_dir():
            # Nasty hack
            if 'heatmap' not in f.path:
                subdirs.append(f.path)

    return subdirs


class FederatedShapDataset(Dataset):
    def __init__(self, model1_path, model2_path, flatten=False):
        super(FederatedShapDataset, self).__init__()

        self.flatten = flatten

        # Each path contains .pkl files, which are the SHAP of the global model
        # And subdirectories, which contain the SHAP values of the local models
        # We will train on the global models values only, the others are used as test
        # There's also no way we can load everything into memory at once, so will have to do it on the fly
        self.ids = []
        self.train_ids = []
        self.labels = []

        # A sample is a dict {global_model_id, local_model, mnist_label, shap_id}
        # global_model can be used as the label
        # The rest can be used to load the SHAP values later on
        samples = []

        # Find the number of local models first
        local_models1 = get_subdirs(model1_path)
        local_models2 = get_subdirs(model2_path)
        global_models = [model1_path, model2_path]
        self.global_models = global_models
        local_models = [local_models1, local_models2]

        # Now find the number of SHAP values
        overall_id = 0
        for label in range(0, 10):
            # For each model
            for model_id in range(len(global_models)):
                model = global_models[model_id]

                # Get the global SHAP vals
                path = os.path.join(model, '{}.pkl'.format(label))

                with open(path, 'rb') as f:
                    shap_vals = pickle.load(f)

                    for i in range(len(shap_vals)):
                        sample = {'global_model_id': model_id, 'local_model': None, 'mnist_label': label, 'shap_id': i}
                        samples.append(sample)

                        self.train_ids.append(overall_id)
                        overall_id += 1

                # Now do the same for each local model
                for local_model in local_models[model_id]:
                    path = os.path.join(str(model), str(local_model), '{}.pkl'.format(label))

                    with open(path, 'rb') as f:
                        shap_vals = pickle.load(f)

                        for i in range(len(shap_vals)):
                            sample = {'global_model_id': model_id, 'local_model': local_model, 'mnist_label': label,
                                      'shap_id': i}
                            samples.append(sample)

                            overall_id += 1

        self.df = pd.DataFrame(samples)

        self.test_ids = list(set([i for i in range(len(samples))]) - set(self.train_ids))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]

        label = rows['global_model_id']

        # Load the correct SHAP value
        if rows['local_model'] is None:
            path = os.path.join(self.global_models[label], '{}.pkl'.format(rows['mnist_label']))
        else:
            path = os.path.join(self.global_models[label], rows['local_model'], '{}.pkl'.format(rows['mnist_label']))

        with open(path, 'rb') as f:
            shap_vals = pickle.load(f)

            shap = shap_vals[rows['shap_id']]

        if not self.flatten:
            # If we want a 2D matrix, but are given a 1D matrix, make it 2D
            if len(shap.shape) == 1:
                # Assume that we can make a square image
                size = int(math.sqrt(shap.shape[0]))
                shap = shap.reshape(size, size)
                shap = shap[None, :, :]

        return shap, label


class IntegratedGradientsDataset(Dataset):
    def __init__(self, path=None, calculate=False, path_prefix='', path_postfix='', flatten=False, model_path=None,
                 return_inputs=False, state_dict=False):
        super(IntegratedGradientsDataset, self).__init__()

        self.flatten = flatten
        self.return_input = return_inputs

        if not calculate:
            if path is None:
                raise Exception('If not calculating IG values, must pass a path')

            # For SHAP, we sometimes have data with labels and sometimes not. For now, assume we have data with labels
            # That means that path is a directory that contains 0.pkl, 1.pkl, etc

            rows = []
            for label in range(0, 10):
                current_path = os.path.join(path, "{}{}{}.pkl".format(path_prefix, label, path_postfix))

                with open(current_path, 'rb') as f:
                    shap_vals = pickle.load(f)

                    for shap in shap_vals:
                        row = [shap, label]
                        rows.append(row)

            self.df = pd.DataFrame(rows, columns=['ig', 'label'])
        else:
            if model_path is None:
                raise Exception('Must pass a model for IG calculations')

            dataset = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print('============= Using device', device)

            if state_dict:
                model = model_path
            else:
                print('============= Loading model from', model_path)
                model = torch.load(model_path)

            model = model.to(device)

            rows = []
            labels = torch.unique(dataset.targets)

            for label in labels:
                if self.return_input:
                    ig_label_samples, inputs = calculate_all_ig_importance(model, dataset, device, multiclass=True,
                                                                               flatten=flatten, target=label,
                                                                               return_inputs=self.return_input)
                else:
                    ig_label_samples = calculate_all_ig_importance(model, dataset, device, multiclass=True,
                                                                       flatten=flatten, target=label,
                                                                       return_inputs=self.return_input)

                for i in range(len(ig_label_samples)):
                    sample = ig_label_samples[i]

                    if self.return_input:
                        img = inputs[i][0]
                        row = [sample, label, img]
                    else:
                        row = [sample, label]

                    rows.append(row)

            if self.return_input:
                self.df = pd.DataFrame(rows, columns=['ig', 'label', 'img'])
            else:
                self.df = pd.DataFrame(rows, columns=['ig', 'label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self.df.iloc[idx]
        ig = torch.tensor(rows['ig'])
        label = torch.tensor(int(rows['label']))

        if not self.flatten:
            # If we want a 2D matrix, but are given a 1D matrix, make it 2D
            if len(ig.shape) == 1:
                # Assume that we can make a square image
                size = int(math.sqrt(ig.shape[0]))
                ig = ig.reshape(size, size)
                ig = ig[None, :, :]

        if self.return_input:
            img = rows['img']
            return ig, label, img
        else:
            return ig, label


if __name__ == '__main__':
    dataset = FederatedShapDataset(model1_path='fed_avg_models/shap/1/', model2_path='fed_avg_models/shap/2/')
    print(dataset[0])