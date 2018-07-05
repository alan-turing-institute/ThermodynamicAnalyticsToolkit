import logging
import sqlite3

from TATi.common import get_list_from_string

class runtime(object):
    """ This class contains runtime information and capability
    to write these to an sqlite file.

    """

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.time_init_network = 0.
        self.time_train_network = 0.
        self.time_overall = 0.

    def set_init_network_time(self, _time):
        self.time_init_network = _time

    def set_train_network_time(self, _time):
        self.time_train_network = _time

    def set_overall_time(self, _time):
        self.time_overall = _time

    def __del__(self):
        if self.FLAGS.sql_db is not None:
            with sqlite3.connect(self.FLAGS.sql_db) as connection:
                cursor = connection.cursor()
                # don't drop anything, just create if not exists
                add_table_command = """
                    CREATE TABLE IF NOT EXISTS run_time (
                    id INTEGER PRIMARY KEY,
                    batch_size INTEGER,
                    dimension INTEGER,
                    hidden_num_layers INTEGER,
                    hidden_min_nodes INTEGER,
                    hidden_max_nodes INTEGER,
                    in_memory_pipeline INTEGER,
                    input_dimension INTEGER,
                    inter_ops_threads INTEGER,
                    intra_ops_threads INTEGER,
                    output_dimension INTEGER,
                    seed INTEGER,
                    step_width FLOAT,
                    init_time FLOAT,
                    train_time FLOAT,
                    overall_time FLOAT
                    );
                """
                logging.debug(add_table_command)
                cursor.execute(add_table_command)
                # add values
                add_values_format = """
                    INSERT INTO run_time
                    (batch_size,dimension,hidden_num_layers,hidden_min_nodes,hidden_max_nodes,in_memory_pipeline, \
                    input_dimension,inter_ops_threads,intra_ops_threads,output_dimension,seed,step_width,init_time, \
                    train_time, overall_time)
                    VALUES ({batch_size}, {dimension}, {hidden_num_layers}, {hidden_min_nodes}, {hidden_max_nodes}, \
                    {in_memory_pipeline}, {input_dimension}, {inter_ops_threads}, {intra_ops_threads}, \
                    {output_dimension}, {seed}, {step_width}, {init_time}, {train_time}, {overall_time});
                """
                hidden_dimension = self.FLAGS.hidden_dimension
                min_nodes = 0
                max_nodes = 0
                if len(hidden_dimension) != 0:
                    min_nodes = min(hidden_dimension)
                    max_nodes = max(hidden_dimension)
                inter_ops_threads = self.FLAGS.inter_ops_threads
                if inter_ops_threads is None:
                    inter_ops_threads = 0
                intra_ops_threads = self.FLAGS.inter_ops_threads
                if intra_ops_threads is None:
                    intra_ops_threads = 0
                try:
                    dataset_dimension = self.FLAGS.dimension
                except AttributeError:
                    dataset_dimension = self.FLAGS.batch_size
                try:
                    step_width = self.FLAGS.step_width
                except AttributeError:
                    step_width = self.FLAGS.learning_rate
                add_values_command = add_values_format.format(
                    batch_size=self.FLAGS.batch_size,
                    dimension=dataset_dimension,
                    hidden_num_layers=len(hidden_dimension),
                    hidden_min_nodes=min_nodes,
                    hidden_max_nodes=max_nodes,
                    in_memory_pipeline=int(self.FLAGS.in_memory_pipeline),
                    input_dimension=self.FLAGS.input_dimension,
                    inter_ops_threads=inter_ops_threads,
                    intra_ops_threads=intra_ops_threads,
                    output_dimension=self.FLAGS.output_dimension,
                    seed=self.FLAGS.seed,
                    step_width=step_width,
                    init_time=self.time_init_network,
                    train_time=self.time_train_network,
                    overall_time=self.time_overall)
                logging.debug(add_values_command)
                cursor.execute(add_values_command)
                connection.commit()
