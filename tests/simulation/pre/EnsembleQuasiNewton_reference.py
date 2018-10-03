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
parser.add_argument('--collapse_walkers', type=int, default=0,
    help='Whether to regularly collapse all dependent walkers to restart from a single position '
         'again, maintaining harmonic approximation for ensemble preconditioning. 0 will never collapse.')
parser.add_argument('--covariance_after_steps', type=int, default=0,
    help='Number of steps after the covariance matrix is recalculated.')
parser.add_argument('--covariance_blending', type=float, default=0.,
    help='Blending between unpreconditioned gradient (0.) and preconditioning through covariance matrix from other '
         'dependent walkers')
parser.add_argument('--every_nth', type=int, default=1,
    help='Store only every nth trajectory (and run) point to files, e.g. 10')
parser.add_argument("--fix_parameters", type=str, default="",
    help="List of parameters to fix (and assign), e.g., output/biases/Variable:0=0.")
parser.add_argument("--friction_constant", type=float, default=None, \
    help="friction constant gamma")
parser.add_argument('--hidden_activation', type=str, default="relu",
    help='Activation function to use for hidden layer: tanh, relu, linear')
parser.add_argument('--hidden_dimension', type=int, nargs='+', default=[],
    help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
parser.add_argument("--input_columns", type=str, nargs='+', default=[], \
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

print("Parameters: "+str(params))

# fix seed
np.random.seed(params.seed)
np.set_printoptions(precision=16)

# setup test pipeline
nn = tati(
    batch_data_files=[params.batch_data_files],
    batch_data_file_type=batch_data_file_type,
    batch_size=None,
    every_nth=100,
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
    verbose=1,
)
nn.non_simplified_access = True  # allow access to parameters using [0] for one walker

if params.trajectory_file is not None:
    tf = open(params.trajectory_file, "w")
    output_width=8
    output_precision=8
    header = get_trajectory_header(
        nn._nn.weights[0].get_total_dof(),
        nn._nn.biases[0].get_total_dof())
    tf.write(",".join(header)+"\n")


def print_step(step, loss_eval, gradients):
    print("Step #" + str(step) + ": " + str(loss_eval) + " at " \
        + str(nn.parameters) + ", gradients " + str(gradients))


def write_trajectory_step(step, gradients):
    if (step % params.every_nth) == 0:
        if params.trajectory_file is not None:
            loss_eval = nn.loss()
            for walker_index in range(params.number_walkers):
                trajectory_line = [str(walker_index), str(step)] \
                  + ['{:{width}.{precision}e}'.format(loss_eval[walker_index], width=output_width,
                      precision=output_precision)] \
                  + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                     for item in nn.parameters[walker_index]]
                tf.write(",".join(trajectory_line)+"\n")
        #print_step(step, loss_eval, gradients)

random_noise_t = []
for walker_index in range(params.number_walkers):
    random_noise_t.append(tensorflow.random_normal(shape=[nn.num_parameters()], mean=0., stddev=1., dtype=tensorflow.float32, seed=params.seed+1+walker_index))

def baoab_update_step(nn, momenta, new_gradients, preconditioner, step_width, beta, gamma, walker_index=0):
    """ Implementation of BAOAB update step using TATi's simulation interface.

    Note:
        Parameters are contained inside nn. For momenta we use
        python variables as the evaluation of the loss does not
        depend on them.

    :param nn: ref to tati simulation instance
    :param momenta: numpy array of parameters
    :param new_gradients: gradients evaluated at last step
    :param preconditioner: preconditioner matrix
    :param step_width: step width for sampling step
    :param beta: inverse temperature
    :param gamma: friction constant
    :param walker_index: index of walker to update
    :return: updated gradients and momenta
    """

    def B(step_width, gradients):
        nonlocal momenta
        momenta = momenta - .5 * step_width * preconditioner.dot(gradients)

    def A(step_width, momenta):
        nn.parameters[walker_index] = nn.parameters[walker_index] + .5 * step_width * preconditioner.dot(momenta)

    def O(step_width, beta, gamma):
        nonlocal momenta
        alpha = math.exp(-gamma * step_width)
        momenta = alpha * momenta + \
                  math.sqrt((1. - math.pow(alpha, 2.)) / beta) * nn._nn.sess.run(random_noise_t[walker_index])

    # 5. p_{n+1} = \widehat{p}_{n+\tfrac 1 2} - \tfrac {\lambda}{2} \nabla_x L(x_{n+1})
    B(step_width, new_gradients)
    #print("B2, #"+str(walker_index)+": "+str(momenta))

    # 1. B: p_{n+\tfrac 1 2} = p_n - \tfrac {\lambda}{2} \nabla_x L(x_n)
    B(step_width, new_gradients)
    #print("B1, #"+str(walker_index)+": "+str(momenta))

    # 2. A: x_{n+\tfrac 1 2} = x_n + \lambda p_{n+\tfrac 1 2}
    A(step_width, momenta)
    #print("A1, #"+str(walker_index)+": "+str(nn.parameters[walker_index]))

    # 3. O: \widehat{p}_{n+1} = \alpha p_{n+\tfrac 1 2} + \sqrt{\frac{1-\alpha^2}{\beta}} \cdot \eta_n
    O(step_width, beta, gamma)
    #print("O, #"+str(walker_index)+": "+str(momenta))

    # 4. A: x_{n+1} = x_{n+\tfrac 1 2} + \lambda \widehat{p}_{n+\tfrac 1 2}
    A(step_width, momenta)
    #print("A2, #"+str(walker_index)+": "+str(nn.parameters[walker_index]))

    return momenta


def calculate_mean(walker_index):
    means = np.zeros((nn.num_parameters()), dtype=np.float32)
    for other_walker_index in range(nn.num_walkers()):
        #print(other_walker_index)
        if walker_index == other_walker_index:
            continue
        #print(nn.parameters)
        means += nn.parameters[other_walker_index]
        #print(means)
        #print(nn.parameters[other_walker_index])
    if nn.num_walkers() > 1:
        means *= 1./(float(nn.num_walkers()) - 1.)
    return means


# ones on diagonal, 1/(dim-1) everywhere else
if nn.num_walkers() > 1:
    normalization = 1./(float(nn.num_walkers()) - 1.)
else:
    normalization = 1.


def calculate_covariance(step, walker_index):
    means = calculate_mean(walker_index=walker_index)
    #print("Means for walker #"+str(walker_index)+" at "+str(step)+": "+str(means))
    covariance = np.zeros((nn.num_parameters(),nn.num_parameters()), dtype=np.float32)
    for other_walker_index in range(nn.num_walkers()):
        #print(other_walker_index)
        if walker_index == other_walker_index:
            continue
        #print(nn.parameters[other_walker_index])
        difference = nn.parameters[other_walker_index] - means
        #print(difference)
        covariance += np.outer(difference, difference)
        #print(covariance)
    covariance *= normalization
    #print("Covariance for walker #" + str(walker_index) + " at "+str(step)+": "
    #      + str(covariance))
    return covariance


preconditioner = [np.identity((nn.num_parameters()), dtype=np.float32) for i in range(params.number_walkers)]

def update_preconditioner(step):
    if (step) % params.covariance_after_steps:
        return
    # calculate covariance matrix for walker_index (i.e. parameters of all other walkers)
    for walker_index in range(nn.num_walkers()):
        #print("walker_index "+str(walker_index))
        preconditioner[walker_index] = \
            params.covariance_blending * calculate_covariance(step, walker_index=walker_index)
        preconditioner[walker_index] += np.identity((nn.num_parameters()), dtype=np.float32)
        #print(preconditioner[walker_index])
        try:
            preconditioner[walker_index] = np.linalg.cholesky(preconditioner[walker_index])
        except np.linalg.linalg.LinAlgError:
            print("Covariance matrix "+str(preconditioner[walker_index])
                  +" was not positive definite.")
            sys.exit(255)
        #print("Preconditioner for walker #"+str(walker_index)+" at "+str(step)+": "
        #      +str(preconditioner[walker_index]))

        #print("TEST: "+str( \
        #    np.dot(preconditioner[walker_index], preconditioner[walker_index].T.conj())))


momenta = [np.zeros((nn.num_parameters()), dtype=np.float32) for i in range(params.number_walkers)]
new_gradients = [np.zeros((nn.num_parameters()), dtype=np.float32) for i in range(params.number_walkers)]


def perform_step():
    new_gradients = nn.gradients()
    for walker_index in range(nn.num_walkers()):
        # perform sampling step with preconditioning
        #print("grad: " + str(new_gradients[walker_index]))
        momenta[walker_index] = baoab_update_step(
            nn, momenta[walker_index], new_gradients[walker_index],
            preconditioner=preconditioner[walker_index],
            step_width=params.step_width,
            beta=params.inverse_temperature, gamma=params.friction_constant, walker_index=walker_index)


def collapse_walkers(step):
    if (step == 0) or (step % params.covariance_after_steps != 0):
        return
    # collapse all walkers to position of first
    if params.collapse_walkers != 0:
        print("Collapsing walker position onto first walker's.")
        for walker_index in range(1, nn.num_walkers()):
            nn.parameters[walker_index] = nn.parameters[0]


for step in range(params.max_steps):
    #print("Current step is "+str(step))
    write_trajectory_step(step, new_gradients)
    update_preconditioner(step)
    perform_step()
    collapse_walkers(step)

if params.trajectory_file is not None:
        tf.close()

