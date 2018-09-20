import argparse
import logging
import math
import numpy as np
import sys
import tensorflow

# get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_data_files", type=str, default=None, \
    help="Input dataset file")
parser.add_argument("--batch_size", type=int, default=None, \
    help="Size of each dataset batch")
parser.add_argument('--collapse_after_steps', type=int, default=10,
    help='Number of steps after which to regularly collapse all dependent walkers to restart from a single position '
         'again, maintaining harmonic approximation for ensemble preconditioning. 0 will never collapse.')
parser.add_argument("--fix_parameters", type=str, default="",
    help="List of parameters to fix (and assign), e.g., output/biases/Variable:0=0.")
parser.add_argument("--friction_constant", type=float, default=None, \
    help="friction constant gamma")
parser.add_argument('--hidden_activation', type=str, default="relu",
    help='Activation function to use for hidden layer: tanh, relu, linear')
parser.add_argument('--hidden_dimension', type=int, nargs='+', default=[],
    help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
parser.add_argument("--input_columns", nargs='+', default="", \
    help="Input columns to use")
parser.add_argument("--input_dimension", type=int, default=784, \
    help="Input dimension of dataset, number of features")
parser.add_argument("--inverse_temperature", type=float, default=None, \
    help="Inverse of temperature")
parser.add_argument("--loss", type=str, default=None, \
    help="loss function to use")
parser.add_argument("--max_steps", type=int, default=None, \
    help="Maximum number of steps")
parser.add_argument("--number_walkers", type=int, default=1, \
    help="Number of parallel walker to use")
parser.add_argument("--output_dimension", type=int, default=10, \
    help="Output dimension of dataset, number of labels")
parser.add_argument("--output_file", type=str, default=None, \
    help="Output CSV file")
parser.add_argument("--parse_parameters_file", type=str, default=None, \
    help="Parse starting parameters from this file")
parser.add_argument("--parse_steps", type=int, nargs='+', default=[], \
    help="step to use from parsed parameter file")
parser.add_argument("--seed", type=int, default=None, \
    help="Seed for random starting configuration")
parser.add_argument("--step_width", type=float, default=None, \
    help="Step width for gradient descent")
parser.add_argument("--tati_path", type=str, default=None, \
    help="Path to tati module")
parser.add_argument("--trajectory_file", type=str, default=None, \
    help="Output trajectory CSV file")
parser.add_argument('--version', '-V', action="store_true", \
    help='Gives version information')

params, _ = parser.parse_known_args()

if params.version:
    # give version and exit
    print(sys.argv[0]+" -- version 0.1")
    sys.exit(0)

sys.path.append(params.tati_path)

from TATi.common import get_trajectory_header
import TATi.simulation as tati


if "csv" in params.batch_data_files:
    batch_data_file_type="csv"
elif "tfrecord" in params.batch_data_files:
    batch_data_file_type="tfrecord"
else:
    print("Unknown input format")
    sys.exit(255)

# fix seed
np.random.seed(params.seed)

# setup test pipeline
nn = tati(
    batch_data_files=[params.batch_data_files],
    batch_data_file_type=batch_data_file_type,
    batch_size=None,
    every_nth=1,
    fix_parameters=params.fix_parameters,
    hidden_activation=params.hidden_activation,
    hidden_dimension=params.hidden_dimension,
    in_memory_pipeline=True,
    input_columns=params.input_columns,
    input_dimension=params.input_dimension,
    loss=params.loss,
    max_steps=params.max_steps,
    number_walkers=params.number_walkers,
    optimizer="GradientDescent",
    output_activation="linear",
    output_dimension=params.output_dimension,
    parse_parameters_file=params.parse_parameters_file,
    parse_steps=params.parse_steps,
    seed=params.seed,
    step_width=params.step_width,
    verbose=2,
)

if params.trajectory_file is not None:
    tf = open(params.trajectory_file, "w")
    output_width=8
    output_precision=8
    header = get_trajectory_header(
        nn._nn.weights[0].get_total_dof(),
        nn._nn.biases[0].get_total_dof())
    tf.write(",".join(header)+"\n")

def write_trajectory_step(step):
    if params.trajectory_file is not None:
        for walker_index in range(params.number_walkers):
            trajectory_line = [str(walker_index), str(step)] \
              + ['{:{width}.{precision}e}'.format(nn.loss()[walker_index], width=output_width,
                  precision=output_precision)] \
              + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                 for item in nn.parameters[walker_index]]
            tf.write(",".join(trajectory_line)+"\n")



def baoab_update_step(nn, momenta, old_gradients, step_width, beta, gamma, walker_index=0):
    """ Implementation of BAOAB update step using TATi's simulation interface.

    Note:
        Parameters are contained inside nn. For momenta we use
        python variables as the evaluation of the loss does not
        depend on them.

    :param nn: ref to tati simulation instance
    :param momenta: numpy array of parameters
    :param old_gradients: gradients evaluated at last step
    :param step_width: step width for sampling step
    :param beta: inverse temperature
    :param gamma: friction constant
    :param walker_index: index of walker to update
    :return: updated gradients and momenta
    """

    # 1. B: p_{n+\tfrac 1 2} = p_n - \tfrac {\lambda}{2} \nabla_x L(x_n)
    momenta -= .5*step_width * old_gradients

    # 2. A: x_{n+\tfrac 1 2} = x_n + \lambda p_{n+\tfrac 1 2}
    nn.parameters[walker_index] = nn.parameters[walker_index] + .5*step_width * momenta

    # 3. O: \widehat{p}_{n+1} = \alpha p_{n+\tfrac 1 2} + \sqrt{\frac{1-\alpha^2}{\beta}} \cdot \eta_n
    alpha = math.exp(-gamma*step_width)
    momenta = alpha * momenta + \
              math.sqrt((1.-math.pow(alpha,2.))/beta) * np.random.standard_normal(momenta.shape)

    # \nabla_x L(x_{n+\tfrac 1 2})
    gradients = nn.gradients()[walker_index]

    # 4. A: x_{n+1} = x_{n+\tfrac 1 2} + \lambda \widehat{p}_{n+\tfrac 1 2}
    nn.parameters[walker_index] = nn.parameters[walker_index] + .5*step_width * momenta

    # 3. p_{n+1} = \widehat{p}_{n+\tfrac 1 2} - \tfrac {\lambda}{2} \nabla_x L(x_{n+1})
    momenta -= .5*step_width * gradients

    return gradients, momenta


momenta = [np.zeros((nn.num_parameters())) for i in range(params.number_walkers)]
old_gradients = [np.zeros((nn.num_parameters())) for i in range(params.number_walkers)]

step=0
print("Step #" + str(step) + ": " + str(nn.loss()) + " at " \
      + str(nn.parameters) + ", gradients " + str(old_gradients))

for leg in range(1): # int(params.max_steps/params.collapse_after_steps)

    for i in range(params.collapse_after_steps):
        step += 1
        for walker_index in range(params.number_walkers):
            # TODO: calculate covariance matrix for walker_index (i.e. parameters of all other walkers)

            old_gradients[walker_index], momenta[walker_index] = baoab_update_step(
                nn, momenta[walker_index], old_gradients[walker_index], step_width=params.step_width,
                beta=params.inverse_temperature, gamma=params.friction_constant, walker_index=walker_index)
        print("Step #"+str(step)+": "+str(nn.loss())+" at " \
            +str(nn.parameters)+", gradients "+str(old_gradients))

        write_trajectory_step(step)

    # collapse all walkers to position of first
    for walker_index in range(1, params.number_walkers):
        nn.parameters[walker_index] = nn.parameters[0]

tf.close()

