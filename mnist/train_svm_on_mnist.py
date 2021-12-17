import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description='Train an SVM for MNIST classification')

    parser.add_argument('--data-path', type=str, help='Path to save/load MNIST to', default='./')
    parser.add_argument('--test-split', type=float, help='Fraction of dataset to reserve for testing', default=0.2)
    parser.add_argument('--seed', type=int, help='Random seed to use', default=1)
    parser.add_argument('--split-seed', type=int, help='Random seed to use when shuffling, if None use same as --seed',
                        default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')

    # From https://github.com/ksopyla/svm_mnist_digit_classification, the default params are good
    parser.add_argument('--c', type=int, help='C parameter for SVM', default=15)
    parser.add_argument('--gamma', type=float, help='Gamma parameter for SVM', default=0.001)

    parser.add_argument('--save-model', type=str, help='Path to save SVM to', default=None)
    parser.add_argument('--save-results', help='Path to save results to', type=str, default=None)

    args = parser.parse_args()

    # Make sure the numpy random seed isn't affecting anything
    np.random.seed(args.seed)

    X_data, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # Split data to train and test
    if args.split_seed is None:
        split_seed = args.seed
    else:
        np.random.seed(args.split_seed)
        split_seed = args.split_seed

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=args.test_split, random_state=split_seed,
                                                        shuffle=args.shuffle)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.random.seed(args.seed)

    print('Training SVM...')
    classifier = svm.SVC(C=args.c, gamma=args.gamma, kernel='rbf', random_state=args.seed, shrinking=True,
                         probability=True)
    classifier.fit(X_train, y_train)

    # Now predict the value of the test
    expected = y_test
    predicted = classifier.predict(X_test)

    classification_report = metrics.classification_report(expected, predicted)
    cm = metrics.confusion_matrix(expected, predicted)
    acc = metrics.accuracy_score(expected, predicted)

    results = "{}\n\n{}\n\n{}".format(classification_report, cm, acc)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, classification_report))

    print("Confusion matrix:\n%s" % cm)

    print("Accuracy={}".format(acc))

    if args.save_model is not None:
        filename = 'mnist-svm-seed{}-splitSeed{}-shuffle{}.pkl'.format(args.seed, split_seed, args.shuffle)
        path = os.path.join(args.save_model, filename)

        print("======================= Saving MNIST SVM to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

    if args.save_results is not None:
        filename = 'mnist-svm-results-seed{}-splitSeed{}-shuffle{}.txt'.format(args.seed, split_seed, args.shuffle)
        path = os.path.join(args.save_results, filename)

        print("======================= Saving results to {}".format(path))

        with open(path, 'w') as f:
            f.write(results)


if __name__ == '__main__':
    main()