import argparse
import numpy as np
from tqdm import tqdm

from ExplanationDatasets import ShapDataset, LimeDataset, LrpDataset, DeepliftDataset
from finetune_densenet import get_cxr_datasets
from utils import VisdomLinePlotter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models

from captum.metrics import infidelity
from captum.attr import GradientShap


global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For now, just add noise as our perturbations
def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0.485, 1, inputs.shape)).float()
    noise = noise.to(device)
    diff = (inputs - noise)
    diff = diff.to(device)

    return noise, diff


def perturb_fn_square(inputs):
    im_size = inputs[0].shape
    print('im_size:', im_size)
    width = im_size[1]
    height = im_size[2]
    rads = np.arange(10) + 1
    num = 0
    for rad in rads:
        num += (width - rad + 1) * (height - rad + 1)

    rangelist = np.arange(np.prod(im_size)).reshape(im_size)
    width = im_size[1]
    height = im_size[2]

    perturbed_images = []
    inds = []
    for image_copy in inputs:
        image_copy = np.tile(np.reshape(np.copy(image_copy.cpu()), -1), [num, 1])

        ind = np.zeros(image_copy.shape)
        count = 0
        for rad in rads:
            for i in range(width - rad + 1):
                for j in range(height - rad + 1):
                    ind[count, rangelist[:, i:i + rad, j:j + rad].flatten()] = 1
                    image_copy[count, rangelist[:, i:i + rad, j:j + rad].flatten()] = 0
                    count += 1

        inds.append(ind)
        perturbed_images.append(image_copy)

    return torch.tensor(inds).to(device), torch.tensor(perturbed_images).to(device)


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy of an explanation method')
    parser.add_argument('--model', type=str, help='Path to original model', required=True)
    parser.add_argument('--method', choices=['shap', 'lime', 'lrp', 'deeplift'], default='shap',
                        help='Explanation method to use')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--pert', choices=['noise', 'square'], default='noise',
                        help='Perturbation to use when calculating infidelity')

    parser.add_argument('--data-path', type=str, default='../data', help='Path to download MIMIC-CXR data to')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use when calculating sensitivity')
    parser.add_argument('--ignore-empty-labels', help='Ignore samples where the diagnosis is not mentioned',
                        action='store_true')

    parser.add_argument('--label', help='Label to classify', type=str, default='Edema')
    parser.add_argument('--checkpoint', action='store_true', help='If loading checkpoint')
    args = parser.parse_args()

    print('============= Using device', device)

    print('============= Loading model from', args.model)
    model = models.densenet121(pretrained=True)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    if args.checkpoint:
        checkpoint = torch.load(args.model)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(args.model))
    model = model.to(device)

    model.eval()

    test_set, _, _ = get_cxr_datasets(args.data_path, args.label, None, args.ignore_empty_labels)

    dataloader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)

    if args.method == 'shap':
        print('============= Using SHAP explanations')
        expl_method = GradientShap(model, multiply_by_inputs=False)
    elif args.method == 'lime':
        print('============= Using LIME explanations')
        dataset = LimeDataset(args.path, return_inputs=True)
    elif args.method == 'lrp':
        print('============= Using LRP explanations')
        dataset = LrpDataset(args.path, return_inputs=True)
    elif args.method == 'deeplift':
        print('============= Using DeepLIFT explanations')
        dataset = DeepliftDataset(args.path, return_inputs=True)
    else:
        raise NotImplementedError()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    infidelities = []
    with tqdm(total=len(dataloader)) as progress:
        for data, targets in dataloader:
            if args.method == 'shap':
                baseline = torch.zeros_like(data)
                baseline[:, 0, :, :] = 0.485
                baseline[:, 1, :, :] = 0.456
                baseline[:, 2, :, :] = 0.406
                baseline = baseline.to(device)
            else:
                baseline = torch.zeros_like(data)
                baseline = baseline.to(device)

            data, targets = data.to(device), targets.to(device)
            attributions = expl_method.attribute(data, baselines=baseline, target=targets)

            if args.pert == 'noise':
                inf = infidelity(model, perturb_fn, data, attributions, target=targets)
            else:
                inf = infidelity(model, perturb_fn_square, data, attributions, target=targets)

            infidelities.append(inf.detach().cpu().flatten())

            progress.update(1)

    infidelities = torch.cat(infidelities)
    print(infidelities)
    print('num_samples:', infidelities.shape)
    print('infid sum:', torch.sum(infidelities))
    print('max:', torch.max(infidelities))
    print('Mean infid:', torch.mean(infidelities))
    print('median infid:', torch.median(infidelities))


if __name__ == '__main__':
    main()