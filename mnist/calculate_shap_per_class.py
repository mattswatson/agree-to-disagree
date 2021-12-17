import argparse
import pickle
import os

from utils import calculate_all_shap_importance
from mnist import Net, SmallNet, MLP, load_model, GaborNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18

def main():
    parser = argparse.ArgumentParser(description='Calculate and save SHAP values for a model on a given dataset')
    parser.add_argument('model', type=str, help='Path to PyTorch model to interpret')
    parser.add_argument('save_path', type=str, help='Path to save calculated SHAP values to')
    parser.add_argument('--data-path', type=str, default='../../data', help='Path to download MNIST data to')
    parser.add_argument('--no-flatten', action='store_true', help='Don\'t flatten the SHAP values')

    parser.add_argument('--state-dict', action='store_true', help='If loading state dict instead of whole model object')
    parser.add_argument('--model-class', choices=['large-cnn', 'small-cnn', 'mlp', 'ensemble', 'gabor', 'resnet18'],
                        default='large-cnn',
                        help='Model to train, only use if passing state dict (default: large-cnn)')

    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'compas', 'diabetes', 'pima', 'regression', 'breast',
                                              'kingdom', 'dna'], default='mnist', help='Dataset to use')

    parser.add_argument('--all-classes', action='store_true',
                        help='Classify all classes. Only used in kingdom, dna (default: keep classes with more than '
                             '1000 samples)')

    parser.add_argument('--baseline', choices=['zero', 'random', 'mean'], default='random',
                        help='Baseline to use for SHAP')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flatten = not args.no_flatten

    print('============= Using device', device)

    if args.dataset == 'mnist':
        dataset = datasets.MNIST(args.data_path, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        mean = 0.1307
    elif args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        mean = 0.5
    elif args.dataset == 'compas':
        dataset = CompasFairMLDataset(args.data_path)

        mean = dataset.whole_dataset_mean()
    elif args.dataset == 'diabetes':
        dataset = DiabetesDatasetLastVisit(args.data_path)

        mean = dataset.mean_sample().type(torch.FloatTensor)
    elif args.dataset == 'pima':
        dataset = PimaDataset(args.data_path)

        mean = dataset.mean_sample().type(torch.FloatTensor)
    elif args.dataset == 'regression':
        dataset = DiabetesRegressionDataset(args.data_path, flatten=True)

        mean = dataset.mean_sample().type(torch.FloatTensor)
    elif args.dataset == 'breast':
        dataset = BreastCancerDataset()

        mean = dataset.whole_dataset_mean().type(torch.FloatTensor)
    elif args.dataset == 'kingdom':
        dataset = KingdomDataset(args.data_path, all_classes=args.all_classes)

        mean = dataset.mean_sample()
    elif args.dataset == 'dna':
        dataset = DNADataset(args.data_path, all_classes=args.all_classes)

        mean = dataset.mean_sample()
    else:
        raise ValueError('Dataset {} not supported!'.format(args.dataset))

    print('============= Loading model from', args.model)
    multiclass = True
    if args.dataset == 'compas' and args.model_class != 'ensemble':
        model = CompasFairMLModel(input_size=dataset.num_features, hidden_size=6).to(device)

        model.load_state_dict(torch.load(args.model))
    elif args.dataset == 'cifar10' and args.model_class != 'ensemble':
        model = CifarNet().to(device)

        model.load_state_dict(torch.load(args.model))
    elif args.dataset == 'diabetes' and args.model_class != 'ensemble':
        model = MLPDiabetes().to(device)

        model.load_state_dict(torch.load(args.model))
    elif args.dataset == 'pima' and args.model_class != 'ensemble':
        model = MLPPima().to(device)

        model.load_state_dict(torch.load(args.model))
    elif args.dataset == 'regression' and args.model_class != 'ensemble':
        model = MLPRegression(dropout1=0.5, dropout2=0.5).to(device)

        model.load_state_dict(torch.load(args.model))
    elif args.dataset == 'breast' and args.model_class != 'ensemble':
        model = MLPPima(num_features=dataset.num_features, hidden_size1=40, hidden_size2=15).to(device)

        model.load_state_dict(torch.load(args.model))
    elif (args.dataset == 'kingdom' or args.dataset == 'dna') and args.model_class != 'ensemble':
        model = MLPGene(input_size=dataset.num_features, num_classes=dataset.num_classes)

        model.load_state_dict(torch.load(args.model))
    elif args.model_class == 'ensemble':
        if args.dataset == 'regression':
            model = get_models_regression(args, model_file=args.model, as_ensemble=True, train=False)
        else:
            model = get_models(args, model_file=args.model, as_ensemble=True, train=False)
    else:
        model = load_model(args.model, args.model_class, args.state_dict, device)
    model = model.to(device)

    if args.baseline == 'mean':
        if type(mean) == int or type(mean) == float:
            sample = dataset[0][0]
            baseline = torch.zeros_like(sample[None, :, :, :])
            baseline = baseline.fill_(mean)
        else:
            baseline = mean[None, :]
    else:
        baseline = args.baseline

    if args.dataset == 'regression':
        shap_label_samples = calculate_all_shap_importance(model, dataset, device, multiclass=False, flatten=flatten,
                                                           baseline=baseline)

        path = os.path.join(args.save_path, '0.pkl')
        with open(path, 'wb') as f:
            print('============= Saving all SHAP values to', path)
            pickle.dump(shap_label_samples, f)
    else:
        labels = torch.unique(torch.tensor(dataset.targets))

        for label in labels:
            shap_label_samples = calculate_all_shap_importance(model, dataset, device, multiclass=multiclass,
                                                               flatten=flatten, target=label, baseline=baseline)

            path = os.path.join(args.save_path, '{}.pkl'.format(label))
            with open(path, 'wb') as f:
                print('============= Saving all SHAP values to', path)
                pickle.dump(shap_label_samples, f)


if __name__ == '__main__':
    main()