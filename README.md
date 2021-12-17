# Agree to Disagree: When Deep Learning Models with Identical Architectures Produce Distinct Explanations

This is the support repository for our paper Agree to Disagree: When Deep Learning Models with Identical Architectures 
Produce Distinct Explanations, M Watson, B Hasan, N Al Moubayed (2021) presented at WACV 2022. Code to re-produce all
experiments outlined in the paper are included, with the code split up into that for MNIST (found in the `mnist`
folder) and MIMIC-CXR (found in the `mimic-cxr` folder).

Note that all model training and SHAP/IG calculation code has CLI options for setting the random seed used. To accurately
reproduce our experiments, one must set the random seed to those reported in the paper. As discussed in the paper, however,
_similar_ results can be obtained when any set of training hyperparemters are used (they just will not be identical to
our reported results).

The code requires a number of Python libraries. Dependencies can be found in `requirements.txt`.

All models are saved as state dicts. All runnable scripts accept a number of CLI arguments which are explainwd through
comments and the use of `python script_name.py -h`.

## MNIST Code

- `calculate_ig_per_class.py` - Takes a saved model and calculates the Integrated Gradient attributions for MNIST.
- `calculate_shap_per_class.py` - Takes a saved model and calculates the SHAP attributions for MNIST.
- `infidelity.py` - Calculates the explanation infidelity for pre-calculated SHAP/IG values
- `mnist.py` - Trains a model on the MNIST dataset. There are many CLI options for this script that can be used to train a wide varity of models
- `model_cca.py` - Takes two models and computes the CCA similarity between specific layers of these models
- `sensitivity.py` - Calculates the explanation sensitivity of pre-calculated SHAP/IG values
- `train_lr.py` - Trains a binary Logistic Regression classifier on two sets of SHAP/IG values from models
- `train_svm_on_mnist.py` - Train an SVM on the MNIST data

As an example of how to use these scripts, here is how to train a ResNet18 model on MNIST:

`python mnist.py --seed 0 --epochs 14 --data-path /path/to/mnist/ --model resnet18 --save ./mnist-resnet18.pth`

The hyperensemble code is taken from the original [hyperparameter ensemble code](https://github.com/google/uncertainty-baselines).
Please follow the setup instructions in that repostiroy before running the below code. 
You will also need [Edward2](https://github.com/google/edward2) - again, setup instructions can be found im the repository.

- `hyper-ensemble/calculate_shap.py` - Calculate the SHAP values for a hyperensemble model
- `hyper-ensemble/calculate_ig.py` - Calculate the IG values for a hyperensemble model
- `hyper-ensemble/train.py` - Train a hyperensemble model on MNIST

## MIMIC-CXR

We use the JPG version of the MIMIC-CXR dataset, which can be found [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

- `averaging.py` - Train a simple ensemble model on the MIMIC-CXR dataset
- `calculate_ig_per_class.py` - Takes a saved model and calculates the Integrated Gradient attributions for MIMIC-CXR.
- `calculate_shap_per_class.py` - Takes a saved model and calculates the SHAP attributions for MIMIC-CXR.
- `explanation_acc.py` - Calculate the explanation accuracy for pre-computed explanations
- `finetune_densenet.py` - Finetune a pre-trained (on ImageNet) DenseNet121 model on MIMIC-CXR
- `infidelity.py` - Calculates the explanation infidelity for pre-calculated SHAP/IG values
- `sensitivity.py` - Calculates the explanation sensitivity of pre-calculated SHAP/IG values
- `model_cca.py` - Takes two models and computes the CCA similarity between specific layers of these models

As an example of how to run these scripts, here is how to finetune a Densenet121 model on MIMIC-CXR:

`python finetune_densenet.py /path/to/mimic-cxr-jpg/ --seed 0 --label Edema`