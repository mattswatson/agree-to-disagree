import argparse
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from shap import KernelExplainer

from sklearn import svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


def calculate_shap_vales_kernelexplainer(model, background_samples, samples, labels, flatten=True):
    print('Setting up SHAP...')
    shap = KernelExplainer(model.predict_proba, background_samples, link='logit')

    #print('samples shape:', samples.shape)
    #print('samples', samples)

    print('Calculating SHAP values...')

    # We only want to keep the SHAP values for the correct classification
    shap_values_pred = []
    with tqdm(total=len(labels)) as progress:
        for i in range(len(labels)):
            label = int(labels[i])
            sample = samples[i]
            shap_values = shap.shap_values(sample, l1_reg='aic')

            shap_values_pred.append(shap_values[label])

            progress.update(1)

    return shap_values_pred

def main():
    parser = argparse.ArgumentParser(description='Calculae SHAP values on an SVM using KernelSHAP')

    parser.add_argument('svm_path', type=str, help='Path to pickled SVM object')

    parser.add_argument('--data-path', type=str, help='Path to load MNIST from', default='./')
    parser.add_argument('--save-shap', type=str,  help='Path to save the SHAP values to', default='./')

    parser.add_argument('--no-flatten', action='store_true', help='Don\'t flatten the SHAP values')
    parser.add_argument('--full-background-set', action='store_true', help='Use the full dataset as background data')

    args = parser.parse_args()

    print('============= Loading SVM from', args.svm_path)
    with open(args.svm_path, 'rb') as f:
        classifier = pickle.load(f)

    # Get the dataset, shouldn't download if it's already there
    X_data, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)
    X_data_df = pd.DataFrame(X_data)

    flatten = not args.no_flatten

    if args.full_background_set:
        background = X_data
    else:
        # A few options here, we will just take the median sample for now (as the dataset is scaled, this should be 0)
        #background = np.array([[0 for i in range(784)]])
        background = X_data_df.median().values.reshape((1, X_data.shape[1]))
        #background = X_data[:5]
        #background = np.array([X_data.mean(axis=0)])
        print(background)
        print('background shape:', background.shape)

    f = lambda x: classifier.predict_proba(x)[:, 1]

    shap_values = calculate_shap_vales_kernelexplainer(classifier, background, X_data, Y, flatten=flatten)
    #shap_values = calculate_shap_vales_kernelexplainer(classifier.decision_function, background, X_data_df.head(n=5), Y[:5], flatten=flatten)

    with open(args.save_shap, 'wb') as f:
        pickle.dump(shap_values, f)

if __name__ == '__main__':
    main()