import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from absl import flags
from absl import app
from uncertainty_baselines.models import hyperbatchensemble_e_factory as e_factory
from uncertainty_baselines.models import HyperBatchEnsembleLambdaConfig as LambdaConfig
from uncertainty_baselines.models import wide_resnet_hyperbatchensemble
import uncertainty_metrics as um
import train
import shap
import pickle
import os
import utils

from tensorflow.compat.v1.keras.backend import get_session

tf.compat.v1.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '/tmp/wide_resnet', 'Checkpoint directory.')
flags.DEFINE_string('save_dir', './tf-shap', 'Directory to save SHAP values to.')

@tf.function
def main(argv):
    tf.compat.v1.disable_eager_execution()
    tf.random.set_seed(train.FLAGS.seed)

    depth = 28
    width = 10
    num_classes = 10

    dict_ranges = {'min': FLAGS.min_l2_range, 'max': FLAGS.max_l2_range}
    ranges = [dict_ranges for _ in range(6)]  # 6 independent l2 parameters
    model_config = {
        'key_to_index': {
            'input_conv_l2_kernel': 0,
            'group_l2_kernel': 1,
            'group_1_l2_kernel': 2,
            'group_2_l2_kernel': 3,
            'dense_l2_kernel': 4,
            'dense_l2_bias': 5,
        },
        'ranges': ranges,
        'test': None
    }
    lambdas_config = LambdaConfig(model_config['ranges'],
                                  model_config['key_to_index'])

    if FLAGS.e_body_hidden_units > 0:
        e_body_arch = '({},)'.format(FLAGS.e_body_hidden_units)
    else:
        e_body_arch = '()'
    e_shared_arch = '()'
    e_activation = 'tanh'
    filters_resnet = [16]
    for i in range(0, 3):  # 3 groups of blocks
        filters_resnet.extend([16 * width * 2 ** i] * 9)  # 9 layers in each block
    # e_head dim for conv2d is just the number of filters (only
    # kernel) and twice num of classes for the last dense layer (kernel + bias)
    e_head_dims = [x for x in filters_resnet] + [2 * num_classes]


    e_models = e_factory(
        lambdas_config.input_shape,
        e_head_dims=e_head_dims,
        e_body_arch=eval(e_body_arch),  # pylint: disable=eval-used
        e_shared_arch=eval(e_shared_arch),  # pylint: disable=eval-used
        activation=e_activation,
        use_bias=FLAGS.e_model_use_bias,
        e_head_init=FLAGS.init_emodels_stddev)

    ds_info = tfds.builder(FLAGS.dataset).info

    print('ds shape:', ds_info.features['image'].shape)
    print('num classes:', num_classes)

    if FLAGS.dataset == 'mnist':
        img_shape = (32, 32, 3)
    else:
        img_shape = ds_info.features['image'].shape

    print('img_shape:', img_shape)

    model = wide_resnet_hyperbatchensemble(
        input_shape=img_shape,
        depth=depth,
        width_multiplier=width,
        num_classes=num_classes,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        config=lambdas_config,
        e_models=e_models,
        l2_batchnorm_layer=FLAGS.l2_batchnorm,
        regularize_fast_weights=FLAGS.regularize_fast_weights,
        fast_weights_eq_contraint=FLAGS.fast_weights_eq_contraint,
        version=2)
    model.compile()
    lambdas_config = LambdaConfig(model_config['ranges'],
                                  model_config['key_to_index'])

    # Initialize Lambda distributions for tuning
    lambdas_mean = tf.reduce_mean(
        train.log_uniform_mean(
            [lambdas_config.log_min, lambdas_config.log_max]))
    lambdas0 = tf.random.normal((FLAGS.ensemble_size, lambdas_config.dim),
                                lambdas_mean,
                                0.1 * FLAGS.ens_init_delta_bounds)
    lower0 = lambdas0 - tf.constant(FLAGS.ens_init_delta_bounds)
    lower0 = tf.maximum(lower0, 1e-8)
    upper0 = lambdas0 + tf.constant(FLAGS.ens_init_delta_bounds)

    log_lower = tf.Variable(tf.math.log(lower0))
    log_upper = tf.Variable(tf.math.log(upper0))
    lambda_parameters = [log_lower, log_upper]  # these variables are tuned

    batch_size = 32
    per_core_batch_size = 32
    train_dataset = utils.load_dataset(
        split=tfds.Split.TRAIN,
        name=FLAGS.dataset,
        batch_size=batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        repeat=True,
        proportion=FLAGS.train_proportion)

    test_dataset = utils.load_dataset(
        split=tfds.Split.TEST,
        name=FLAGS.dataset,
        batch_size=batch_size,
        use_bfloat16=FLAGS.use_bfloat16)

    X, y = next(train_dataset)

    X = tf.tile(X, [FLAGS.ensemble_size, 1, 1, 1])
    print('x shape:', X.shape)

    # generate lambdas
    lambdas = train.log_uniform_sample(
        per_core_batch_size, lambda_parameters)
    lambdas = tf.reshape(
        lambdas,
        (FLAGS.ensemble_size * per_core_batch_size, lambdas_config.dim))
    print('lambdas shape:', lambdas.shape)

    to_pass = [X, lambdas]
    out = tf.convert_to_tensor(model(to_pass))

    print('out:', out)
    print('out type:', type(out))
    print('out graph:', out.graph)
    explainer = shap.GradientExplainer(model, to_pass)

    test_iterator = iter(test_dataset)
    X, y = next(test_iterator)

    shap_values = explainer.shap_values([X.numpy(), lambdas.numpy()], nsamples=10)

    path = os.path.join(FLAGS.save_dir, 'shap.pkl')

    with open(path, 'wb') as f:
        pickle.dump(shap_values, f)
        print('Saved SHAP vals to {}'.format(path))

if __name__ == '__main__':
    app.run(main)
