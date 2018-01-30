import logging
import sqlite3

from DataDrivenSampler.common import get_list_from_string

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
                    (batch_size,dimension,hidden_num_layers,hidden_min_nodes,hidden_max_nodes,seed,step_width,init_time,train_time, overall_time)
                    VALUES ({batch_size}, {dimension}, {hidden_num_layers}, {hidden_min_nodes}, {hidden_max_nodes}, {seed}, {step_width}, {init_time}, {train_time}, {overall_time});
                """
                hidden_dimension = [int(i) for i in get_list_from_string(self.FLAGS.hidden_dimension)]
                min_nodes = 0
                max_nodes = 0
                if len(hidden_dimension) != 0:
                    min_nodes = min(hidden_dimension)
                    max_nodes = max(hidden_dimension)
                add_values_command = add_values_format.format(
                    batch_size=self.FLAGS.batch_size,
                    dimension=self.FLAGS.dimension,
                    hidden_num_layers=len(hidden_dimension),
                    hidden_min_nodes=min_nodes,
                    hidden_max_nodes=max_nodes,
                    seed=self.FLAGS.seed,
                    step_width=self.FLAGS.step_width,
                    init_time=self.time_init_network,
                    train_time=self.time_train_network,
                    overall_time=self.time_overall)
                logging.debug(add_values_command)
                cursor.execute(add_values_command)
                connection.commit()
