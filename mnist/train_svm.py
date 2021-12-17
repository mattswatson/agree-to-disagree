import argparse
import pickle

from utils import train_binary_svm


arg_parser = argparse.ArgumentParser(description='Train SVMs to detect anonymised data')

arg_parser.add_argument('shap_vals_pos', help='Path to first set of SHAP values', type=str)
arg_parser.add_argument('shap_vals_neg', help='Path to second set of SHAP values', type=str)

arg_parser.add_argument('--kernel', help='Specific kernel to use', type=str, choices=['linear', 'poly', 'rbf',
                                                                                      'sigmoid'], default=None)

arg_parser.add_argument('--test-split', help='Proportion of data to use for testing', type=float, default=0.3)

arg_parser.add_argument('--save', help='Path to save best model to', type=str, default=None)
arg_parser.add_argument('--save-results', help='Path to save results to', type=str, default=None)
arg_parser.add_argument('--results-name', help='Append to name of results file', type=str, default='')

args = arg_parser.parse_args()

with open(args.shap_vals_pos, 'rb') as f:
    shap_vals_pos = pickle.load(f)
    print('=============== Loaded SHAP values from', args.shap_vals_pos)

with open(args.shap_vals_neg, 'rb') as f:
    shap_vals_neg = pickle.load(f)
    print('=============== Loaded SHAP values from', args.shap_vals_neg)

# Make array if needed
if type(shap_vals_neg) != list:
    shap_vals_neg = shap_vals_neg.tolist()

if type(shap_vals_pos) != list:
    shap_vals_pos = shap_vals_pos.tolist()

print("======================= Training SVMs")
train_binary_svm(shap_vals_pos, shap_vals_neg, kernel=args.kernel, save=args.save, save_results=args.save_results,
          test_split=args.test_split, results_name=args.results_name)
