import argparse
import numpy as np
from tqdm import tqdm

from ExplanationDatasets import ShapDataset, LimeDataset, LrpDataset, DeepliftDataset
from mnist import Net, SmallNet, MLP, train, test, load_model, GaborNet
from utils import VisdomLinePlotter

from compas import CompasFairMLModel
from CompasDataset import CompasFairMLDataset

from diabetes import MLPCurrent as MLPDiabetes
from diabetes import MLPPima, MLPRegression
from DiabetesDataset import DiabetesDatasetLastVisit, PimaDataset, DiabetesRegressionDataset
from CancerDataset import BreastCancerDataset
from GeneDatasets import KingdomDataset, DNADataset

from gene import MLP as MLPGene

from explanation_ensemble import get_models, CifarNet
from explanation_ensemble_regression import get_models as get_models_regression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.models import resnet18

from captum.metrics import sensitivity_max
from captum.metrics import infidelity_perturb_func_decorator
from captum.attr import GradientShap, IntegratedGradients


global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For now, just add noise as our perturbations
def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0.1307, 1, inputs.shape)).float()
    noise = noise.to(device)
    diff = (inputs - noise)
    diff = diff.to(device)

    return noise, diff


@infidelity_perturb_func_decorator(True)
def perturb_fn_square(inputs, baselines):
    perturbed_inputs = torch.zeros_like(inputs)
    original_im_size = inputs[0].shape

    for i in range(0, inputs.shape[0]):
        t = inputs[i].flatten()
        im_size = t.shape
        mask = torch.randint(low=0, high=2, size=im_size, device=device)

        t[mask] = 0
        t = torch.reshape(t, original_im_size)
        perturbed_inputs[i] = t

    return perturbed_inputs


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy of an explanation method')
    parser.add_argument('--model', type=str, help='Path to original model', required=True)
    parser.add_argument('--method', choices=['shap', 'lime', 'lrp', 'deeplift', 'ig'], default='shap',
                        help='Explanation method to use')

    parser.add_argument('--calculate-shap', action='store_true', help='If using SHAP, calculate SHAP on the fly rather'
                                                                      'than using saved values')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--pert', choices=['noise', 'square'], default='noise',
                        help='Perturbation to use when calculating infidelity')

    parser.add_argument('--state-dict', action='store_true', help='If loading state dict instead of whole model object')
    parser.add_argument('--model-class', choices=['large-cnn', 'small-cnn', 'mlp', 'gabor', 'resnet18', 'ensemble'],
                        default='large-cnn',
                        help='Model to train, only use if passing state dict (default: large-cnn)')

    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MNIST data to')
    #parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use when calculating sensitivity')

    parser.add_argument('--threshold', type=float, default=None, help='Thresholds to use for explanations')
    parser.add_argument('--kde-step', type=float, default=0.00001)

    parser.add_argument('--save', type=str, default=None, help='Path to save results to')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'compas', 'diabetes', 'pima', 'regression', 'breast',
                                              'kingdom', 'dna'], default='mnist', help='Dataset to use')

    args = parser.parse_args()

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
        dataset = KingdomDataset(args.data_path, all_classes=False)

        mean = dataset.mean_sample()
    elif args.dataset == 'dna':
        dataset = DNADataset(args.data_path, all_classes=False)

        mean = dataset.mean_sample()
    else:
        raise ValueError('Dataset {} not supported!'.format(args.dataset))

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

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
        multiclass = False
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
            multiclass = False
            model = get_models_regression(args, model_file=args.model, as_ensemble=True, train=False)
        else:
            model = get_models(args, model_file=args.model, as_ensemble=True, train=False)
    else:
        model = load_model(args.model, args.model_class, args.state_dict, device)
    model = model.to(device)

    if args.method == 'shap':
        print('============= Using SHAP explanations')
        expl_method = GradientShap(model)
    elif args.method == 'lime':
        print('============= Using LIME explanations')
        raise NotImplementedError()
    elif args.method == 'lrp':
        print('============= Using LRP explanations')
        raise NotImplementedError()
    elif args.method == 'deeplift':
        print('============= Using DeepLIFT explanations')
        raise NotImplementedError()
    elif args.method == 'ig':
        print('============= Using IG explanations')
        expl_method = IntegratedGradients(model)
    else:
        raise NotImplementedError()

    def expl_method_threshold(inputs, baselines=None, target=None, thresh=None):
        orig_expl_method = expl_method

        if multiclass:
            attributions = orig_expl_method.attribute(inputs, baselines=baselines, target=target)[0]
        else:
            attributions = orig_expl_method.attribute(inputs, baselines=baselines)[0]

        # Nasty hack...
        if 0 in attributions:
            return 0

        if thresh is not None:

            from scipy.stats import gaussian_kde

            kde = gaussian_kde(attributions.cpu().numpy().flatten())

            thresh_val = 0
            score = kde.integrate_box_1d(-np.inf, thresh_val)

            while score < thresh:
                thresh_val += args.kde_step

                score = kde.integrate_box_1d(-np.inf, thresh_val)

                if args.verbose:
                    print('thresh val: {}\nprob: {}\n----------\n'.format(thresh_val, score))

            attributions[attributions < thresh_val] = 0

        return attributions

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    all_sens = []
    with tqdm(total=len(dataloader)) as progress:
        for data, targets in dataloader:
            if args.method == 'shap':
                if type(mean) == int:
                    baseline = torch.zeros_like(data)
                    baseline = baseline.fill_(mean)
                    baseline = baseline.to(device)
                else:
                    baseline = mean[None, :].to(device)
            else:
                baseline = torch.zeros_like(data)
                baseline = baseline.to(device)

            data, targets = data.to(device), targets.to(device)

            if args.threshold is None:
                sens_max = sensitivity_max(expl_method.attribute, data, baselines=baseline, target=targets)
            else:
                sens_max = sensitivity_max(expl_method_threshold, data, baselines=baseline, target=targets,
                                           thresh=args.threshold)

            all_sens.append(sens_max.flatten())

            progress.update(1)

    all_sens = torch.cat(all_sens)
    print(all_sens)
    print('num_samples:', all_sens.shape)
    print('sens sum:', torch.sum(all_sens))
    print('max:', torch.max(all_sens))
    print('Mean sens_max:', torch.mean(all_sens))
    print('median sens_max:', torch.median(all_sens))

    if args.save is not None:
        to_save = 'mean sens_max: {}\nmedian sens_max: {}'.format(torch.mean(all_sens), torch.median(all_sens))

        with open(args.save, 'w') as f:
            f.write(to_save)

        print('saved to {}'.format(args.save))


if __name__ == '__main__':
    main()