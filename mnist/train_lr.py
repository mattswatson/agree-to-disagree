import argparse
import pickle
import os

from utils import train_lr, train_lda, train_binary_svm
from ExplanationDatasets import LimeDataset, DeepliftDataset

from torch.utils.data import DataLoader

arg_parser = argparse.ArgumentParser(description='Train LR to detect models data from explanations')

arg_parser.add_argument('--method', choices=['shap', 'lime', 'lrp', 'deeplift', 'deeplift-class', 'ig'], default='shap',
                        help='Explanation method to use')

arg_parser.add_argument('shap_vals_pos', help='Path to first set of SHAP/IG values', type=str)
arg_parser.add_argument('shap_vals_neg', help='Path to second set of SHAP/IG values', type=str)
arg_parser.add_argument('--append-filename', help='Text to append to SHAP filenames', type=str, default='')

arg_parser.add_argument('--test-split', help='Proportion of data to use for testing', type=float, default=0.3)
arg_parser.add_argument('--cv', help='Use cross validation to get best hyperparams', action='store_true')
arg_parser.add_argument('--no-scale', action='store_true', help='Don\'t scale data for LR')

arg_parser.add_argument('--save', help='Path to save best model to', type=str, default=None)
arg_parser.add_argument('--save-results', help='Path to save results to', type=str, default=None)
arg_parser.add_argument('--results-name', help='Append to name of results file', type=str, default='')

arg_parser.add_argument('--n-jobs', help='Number of cores to use if using CV', type=int, default=1)
arg_parser.add_argument('--verbose', help='Level of verbosity', type=int, default=0)

arg_parser.add_argument('--model', choices=['lr', 'lda', 'svm'], default='lr', help='Model to use')

arg_parser.add_argument('--label', help='If using LIME, this is the label to check', type=int, default=None)

arg_parser.add_argument('--old-python', action='store_true', help='Use if SHAP was calculated with Python 2.7')

args = arg_parser.parse_args()

scale = not args.no_scale

if args.method == 'shap' or args.method == 'ig':
    shap_val_pos_filename = args.shap_vals_pos[:-4] + args.append_filename + '.pkl'
    shap_val_neg_filename = args.shap_vals_neg[:-4] + args.append_filename + '.pkl'

    if args.old_python:
        # Filename should be label
        label = os.path.basename(args.shap_vals_pos)[0]

        with open(args.shap_vals_pos, 'rb') as f:
            shap_vals_pos = pickle.load(f, encoding='latin1')
            print('=============== Loaded SHAP/IG values from', args.shap_vals_pos)

            shap_vals_pos = shap_vals_pos[int(label)]
            shap_vals_pos = [s.flatten() for s in shap_vals_pos]

        with open(args.shap_vals_neg, 'rb') as f:
            shap_vals_neg = pickle.load(f, encoding='latin1')
            print('=============== Loaded SHAP/IG values from', args.shap_vals_neg)

            shap_vals_neg = shap_vals_neg[int(label)]
            shap_vals_neg = [s.flatten() for s in shap_vals_neg]
    else:
        with open(shap_val_pos_filename, 'rb') as f:
            shap_vals_pos = pickle.load(f)
            print('=============== Loaded SHAP/IG values from', args.shap_vals_pos)

        with open(shap_val_neg_filename, 'rb') as f:
            shap_vals_neg = pickle.load(f)
            print('=============== Loaded SHAP/IG values from', args.shap_vals_neg)

    # Make array if needed
    if type(shap_vals_neg) != list:
        shap_vals_neg = shap_vals_neg.tolist()

    if type(shap_vals_pos) != list:
        shap_vals_pos = shap_vals_pos.tolist()
elif args.method == 'deeplift':
    if args.label is None:
        raise Exception('Must provide a label!')

    pos_dataset = DeepliftDataset(path=args.shap_vals_pos, flatten=True)
    pos_dataloader = DataLoader(pos_dataset, batch_size=1)
    neg_dataset = DeepliftDataset(path=args.shap_vals_neg, flatten=True)
    neg_dataloader = DataLoader(neg_dataset, batch_size=1)

    shap_vals_pos = []
    shap_vals_neg = []

    for sample, label in pos_dataloader:
        if label[0] == args.label:
            shap_vals_pos.append(sample[0].numpy())

    for sample, label in neg_dataloader:
        if label[0] == args.label:
            shap_vals_neg.append(sample[0].numpy())

    print('pos len:', len(shap_vals_pos))
    print('neg len:', len(shap_vals_neg))
    print('sample shape:', shap_vals_pos[0].shape)
elif args.method == 'deeplift-class':
    with open(args.shap_vals_pos, 'rb') as f:
        shap_vals_pos = pickle.load(f)
        print('=============== Loaded DeepLIFT values from', args.shap_vals_pos)

    with open(args.shap_vals_neg, 'rb') as f:
        shap_vals_neg = pickle.load(f)
        print('=============== Loaded DeepLIFT values from', args.shap_vals_neg)
elif args.method == 'lime':
    if args.label is None:
        raise Exception('Must provide a label!')

    pos_dataset = LimeDataset(path=args.shap_vals_pos, flatten=True)
    pos_dataloader = DataLoader(pos_dataset, batch_size=1)
    neg_dataset = LimeDataset(path=args.shap_vals_neg, flatten=True)
    neg_dataloader = DataLoader(neg_dataset, batch_size=1)

    shap_vals_pos = []
    shap_vals_neg = []

    for sample, label in pos_dataloader:
        if label[0] == args.label:
            shap_vals_pos.append(sample[0].numpy())

    for sample, label in neg_dataloader:
        if label[0] == args.label:
            shap_vals_neg.append(sample[0].numpy())

    print('pos len:', len(shap_vals_pos))
    print('neg len:', len(shap_vals_neg))
    print('sample shape:', shap_vals_pos[0].shape)
else:
    raise NotImplementedError()

if args.model == 'lda':
    print("======================= Training LDAs")
    train_lda(shap_vals_pos, shap_vals_neg, save=args.save, save_results=args.save_results,
              test_split=args.test_split, results_name=args.results_name, n_jobs=args.n_jobs,
              verbose=args.verbose, scale=scale)
elif args.model == 'svm':
    print("======================= Training SVMs")
    train_binary_svm(shap_vals_pos, shap_vals_neg, kernel='rbf', save=args.save, save_results=args.save_results,
                     test_split=args.test_split, results_name=args.results_name, scale=scale)

else:
    print("======================= Training LRs")
    train_lr(shap_vals_pos, shap_vals_neg, save=args.save, save_results=args.save_results,
             test_split=args.test_split, results_name=args.results_name, cv=args.cv, n_jobs=args.n_jobs,
             verbose=args.verbose, scale=scale)

