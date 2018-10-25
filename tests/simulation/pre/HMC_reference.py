import argparse
import logging
import numpy as np
import sys
import tensorflow

from TATi.common import get_trajectory_header
import TATi.simulation as tati

# get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_data_files", type=str, default=None, \
	help="Input dataset file")
parser.add_argument("--batch_size", type=int, default=None, \
	help="Size of each dataset batch")
parser.add_argument("--hamiltonian_dynamics_steps", type=int, default=None, \
	help="Number of hamiltonian dynamics steps")
parser.add_argument("--input_dimension", type=int, default=784, \
	help="Input dimension of dataset, number of features")
parser.add_argument("--inverse_temperature", type=float, default=None, \
	help="Inverse of temperature")
parser.add_argument("--loss", type=str, default=None, \
	help="loss function to use")
parser.add_argument("--max_steps", type=int, default=None, \
	help="Maximum number of steps")
parser.add_argument("--output_dimension", type=int, default=10, \
	help="Output dimension of dataset, number of labels")
parser.add_argument("--output_file", type=str, default=None, \
	help="Output CSV file")
parser.add_argument("--parameter_file", type=str, default=None, \
	help="Parse starting parameters from this file")
parser.add_argument("--seed", type=int, default=None, \
	help="Seed for random starting configuration")
parser.add_argument("--step_width", type=float, default=None, \
	help="Step width for gradient descent")
parser.add_argument("--trajectory_file", type=str, default=None, \
	help="Output trajectory CSV file")
parser.add_argument('--version', '-V', action="store_true", \
	help='Gives version information')

params, _ = parser.parse_known_args()

if params.version:
	# give version and exit
	print(sys.argv[0]+" -- version 0.1")
	sys.exit(0)

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
	fix_parameters="output/biases/Variable:0=0.",
	in_memory_pipeline=True,
	input_dimension=params.input_dimension,
	loss=params.loss,
	max_steps=params.max_steps,
	optimizer="GradientDescent",
	output_activation="linear",
	output_dimension=params.output_dimension,
	seed=params.seed,
	step_width=params.step_width,
	verbose=2,
)

# setup a fixed dict to use
test_dict = {
	nn._nn.xinput: nn.dataset[0],
	nn._nn.nn[0].placeholder_nodes["y_"]: nn.dataset[1],
	nn._nn.nn[0].placeholder_nodes["learning_rate"]: nn._options.step_width,
}

def get_energy(step, momenta):
	loss = nn.loss()
	kinetic_energy = .5*momenta*momenta
	logging.info("L_"+str(step)+" = "+str(loss)+", T_"+str(step)+" = "+str(kinetic_energy)+", H_"+str(step)+" = "+str(kinetic_energy+loss))
	return loss + kinetic_energy

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
		trajectory_line = [str(0), str(step)] \
		  + ['{:{width}.{precision}e}'.format(nn.loss(), width=output_width,
			  precision=output_precision)] \
		  + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
			 for item in nn.parameters]
		tf.write(",".join(trajectory_line)+"\n")

rejections=0
accepts=0

p_old = nn.parameters
write_trajectory_step(0)

accept_seed = int(np.random.uniform(low=0,high=67108864))
#print(accept_seed)
random_momenta_state = np.random.RandomState(seed=params.seed+1)
random_p_accept_state = np.random.RandomState(seed=accept_seed)

random_noise_t = tensorflow.random_normal(shape=[nn.num_parameters()], mean=0., stddev=1., dtype=tensorflow.float32, seed=params.seed+1)
uniform_random_t = tensorflow.random_uniform(shape=[], minval=0., maxval=1., dtype=tensorflow.float32, seed=accept_seed)

p_accept = nn._nn.sess.run(uniform_random_t)
#print(p_accept)

for step in range(params.max_steps):
	momenta = nn._nn.sess.run(random_noise_t) * np.sqrt(1.0 / params.inverse_temperature)
	for i in range(params.hamiltonian_dynamics_steps+1):
		nn._nn.sess.run(random_noise_t)
	logging.debug("q_"+str(step)+" = "+str(p_old)+", p_"+str(step)+" = "+str(momenta))

	H_old = get_energy(step, momenta)

	grad = nn.gradients()

	step_width = np.random.uniform(low=0.7, high=1.3) * params.step_width
	# random is used twice?
	np.random.uniform(low=0.7, high=1.3)
	print(step_width)
	step += 1
	for i in range(params.hamiltonian_dynamics_steps):
		logging.debug("g_{"+str(step)+","+str(i-1)+"} = "+str(grad))
		momenta = momenta - 0.5 * grad * step_width
		logging.debug("B1: p_{"+str(step)+","+str(i-1)+"} = "+str(momenta))
		nn.parameters = nn.parameters + step_width * momenta
		logging.debug("A: q_{"+str(step)+","+str(i-1)+"} = "+str(nn.parameters))
		grad = nn.gradients()
		logging.debug("g_{"+str(step)+","+str(i-1)+"} = "+str(grad))
		momenta = momenta - 0.5 * grad * step_width
		logging.debug("B2: p_{"+str(step)+","+str(i-1)+"} = "+str(momenta))
	logging.debug("proposed q_"+str(step)+", p_"+str(step)+" = "+str(nn.parameters)+", "+str(momenta))

	H_new = get_energy(step, momenta)

	acceptanceProba =  np.min([1, np.exp(-params.inverse_temperature * (H_new - H_old))])
	for i in range(params.hamiltonian_dynamics_steps+1+1):
		p_accept = nn._nn.sess.run(uniform_random_t)
	#print(str(acceptanceProba)+" < "+str(p_accept))
	if acceptanceProba < p_accept:
		# reject
		#theta = theta_old # not needed
		nn.parameters = p_old # -1 not needed
		#increase the rejection counter
		rejections += 1
		logging.info("REJECTED q_"+str(step)+", p_"+str(step)+" = "+str(nn.parameters)+", "+str(momenta))
	else:
		p_old = nn.parameters
		accepts += 1 
		logging.info("ACCEPTED q_"+str(step)+", p_"+str(step)+" = "+str(nn.parameters)+", "+str(momenta))

	write_trajectory_step(step)

tf.close()
print("Accepts: "+str(accepts)+", rejects: "+str(rejections))

