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
from tf_explain.core.integrated_gradients import IntegratedGradients
import pickle
import os
import utils
from tqdm import tqdm

tf.config.run_functions_eagerly(True)

flags.DEFINE_string('checkpoint_dir', './output/', 'Checkpoint directory.')
flags.DEFINE_string('save_dir', './tf-ig', 'Directory to save IG values to.')
flags.DEFINE_integer('n_steps', 5, 'Number of steps in the path for IG')
flags.DEFINE_integer('n_samples', None, 'Number of samples to explain (None: all)')

FLAGS = flags.FLAGS

@tf.function
def main(argv):
    tf.compat.v1.enable_eager_execution()
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
    num_train_samples = ds_info.splits['train'].num_examples

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

    batch_size = 1
    per_core_batch_size = 1
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

    explainer = IntegratedGradients()

    ig_values = {str(i): [] for i in range(10)}
    n_steps = FLAGS.n_steps
    total = FLAGS.n_samples if FLAGS.n_samples is not None else num_train_samples
    with tqdm(total=total) as progress:
        for X, y in train_dataset:
            X = explainer.generate_interpolations(X.numpy(), n_steps=n_steps)
            X = tf.tile(X, [FLAGS.ensemble_size, 1, 1, 1])

            # generate lambdas
            lambdas = train.log_uniform_sample(
                per_core_batch_size * n_steps, lambda_parameters)
            lambdas = tf.reshape(
                lambdas,
                (FLAGS.ensemble_size * per_core_batch_size * n_steps, lambdas_config.dim))

            with tf.GradientTape() as tape:
                inputs = tf.cast(X, tf.float32)
                tape.watch(inputs)
                predictions = model([inputs, lambdas])
                loss = predictions[:, int(y)]

            grads = tape.gradient(loss, inputs)
            grads_per_image = tf.reshape(grads, (-1, 5, *grads.shape[1:]))

            integrated_gradients = tf.reduce_mean(grads_per_image, axis=1)
            print('y', len(y))
            for i in range(len(y)):
                print('i', i)
                label = tf.strings.as_string(y[i]).numpy().decode('utf-8')[0]
                ig_values[label].append(integrated_gradients[i, :, :, :])

            progress.update(1)

            if FLAGS.n_samples is not None:
                count = sum([len(ig_values[x]) for x in ig_values])
                if count > FLAGS.n_samples:
                    break

    ig_values = {x: tf.stack(ig_values[x]) for x in ig_values}

    for i in range(10):
        path = os.path.join(FLAGS.save_dir, 'ig-{}.pkl'.format(i))

        with open(path, 'wb') as f:
            pickle.dump(ig_values[str(i)], f)
            print('Saved IG vals to {}'.format(path))

if __name__ == '__main__':
    app.run(main)
