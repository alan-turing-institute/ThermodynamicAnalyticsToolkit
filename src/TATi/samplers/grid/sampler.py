import numpy as np
import pandas as pd

from TATi.common import setup_csv_file

class Sampler(object):
    """ This class contains general functions to access the neural network in
    an abstract fashion. It serves as the interface to all deriving grid-based
    samplers.

    Samplers are executed in the following loop:
    1. `goto_start()`
    2. `set_step()`
    3. `evaluate_loss()`
    3. `write_output_line()`
    4. `goto_next_step()` (go to step 2.)

    """
    def __init__(self, network_model, exclude_parameters):
        self.network_model = network_model
        self.exclude_parameters = exclude_parameters

        self.nn_weights = network_model.weights[0]
        self.nn_biases = network_model.biases[0]

        # initially set weights and biases to zero (possibly overridden by
        # parse_parameters_file)
        self.weights_vals = self.nn_weights.create_flat_vector()
        self.biases_vals = self.nn_biases.create_flat_vector()

        self._prepare_nodes()

    def assign_values_from_file(self, filename, step_nr, do_check=True):
        trajectory = pd.read_csv(filename, sep=',', header=0)
        rownr = trajectory.index[trajectory.loc[:,'step'].values == step_nr][0]
        weights_eval, biases_eval = self.network_model.assign_weights_and_biases_from_dataframe(
            df_parameters=trajectory,
            rownr=rownr,
            do_check=do_check
        )
        return weights_eval, biases_eval

    def assign_values(self, do_check=True):
        weights_eval, biases_eval = self.network_model.assign_weights_and_biases( \
            self.weights_vals, self.biases_vals, do_check=do_check)
        return weights_eval, biases_eval

    def _prepare_nodes(self):
        self.loss = self.network_model.nn[0].get_list_of_nodes(["loss"])
        self.acc = self.network_model.nn[0].get_list_of_nodes(["accuracy"])
        self._sess = self.network_model.sess

    def _prepare_header(self):
        ## set up output csv file
        return ["step", "loss", "accuracy"]

    def prepare_csv_output(self, filename):
        if filename is not None:
            header = self._prepare_header()
            csv_writer, csv_file = setup_csv_file(filename, header)
            return csv_writer, csv_file
        else:
            return None, None

    def _add_all_degree_header(self, header):
        for i in range(self.weights_vals.size):
            header.append("w" + str(i))
        for i in range(self.biases_vals.size):
            header.append("b" + str(i))
        return header

    def evaluate_loss(self):
        # get next batch of data
        features, labels = self.network_model.input_pipeline.next_batch(self._sess)

        # place in feed dict
        feed_dict = {
            self.network_model.xinput: features,
            self.network_model.nn[0].placeholder_nodes["y_"]: labels
        }

        loss_eval, acc_eval = self._sess.run([self.loss, self.acc], feed_dict=feed_dict)

        #print("Loss and accuraccy at the given parameters w("+str(weights_eval)+" b("
        #      +str(biases_eval)+") is "+str(loss_eval[0])+" and "+str(acc_eval[0]))

        return loss_eval, acc_eval

    def write_output_line(self, csv_writer, loss_eval, acc_eval, coords_eval,
                          output_precision, output_width):
        if csv_writer is not None:
            print_row = [self.current_step] \
                        + ['{:{width}.{precision}e}'.format(loss_eval[0], width=output_width,
                                                            precision=output_precision)] \
                        + ['{:{width}.{precision}e}'.format(acc_eval[0], width=output_width,
                                                            precision=output_precision)] \
                        + ['{:{width}.{precision}e}'.format(coord, width=output_width,
                                                            precision=output_precision)
                                                            for coord in coords_eval[:]]
            csv_writer.writerow(print_row)

    def _combine_into_coords(self, weights_eval, biases_eval):
        return np.concatenate([weights_eval, biases_eval]) \
            if (weights_eval.size != 0) and (biases_eval.size != 0) else \
            (weights_eval if weights_eval.size != 0 else biases_eval)

    def goto_start(self):
        self.current_step = 0

    def set_step(self):
        raise NotImplementedError("Sampler's set_step() cannot be used directly, needs to be overwritten in deriving class.")

    def goto_next_step(self):
        self.current_step += 1
        return None