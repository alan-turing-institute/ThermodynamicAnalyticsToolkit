import csv


def get_list_from_string(str_or_list_of_str):
    """ Extracts list of ints from any string (or list of strings).

    :param str_or_list_of_str: string
    :return: list of ints
    """
    tmpstr=str_or_list_of_str
    if str_or_list_of_str is not str:
        try:
            tmpstr=" ".join(str_or_list_of_str)
        except(TypeError):
            tmpstr=" ".join([item for sublist in str_or_list_of_str for item in sublist])
    return [int(item) for item in tmpstr.split()]


def initialize_config_map():
    """ This initialize the configuration dictionary with default values

    :return:
    """
    config_map = {}

    # output files
    config_map["do_write_csv_file"] = False
    config_map["csv_file"] = None
    config_map["do_write_trajectory_file"] = False
    config_map["trajectory_file"] = None

    return config_map


def setup_csv_file(filename, header):
    """ Opens a new CSV file and writes the given `header` to it.

    :param filename: filename of CSV file
    :param header: header to write as first row
    :return: csv writer, csv file
    """
    csv_file = open(filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)
    return csv_writer, csv_file


def setup_run_file(filename, header, config_map):
    """ Opens the run CSV file if a proper `filename` is given.

    :param filename: filename of run CSV file or None
    :param header: list of strings as header for each column
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_csv_file"] = True
        csv_writer, config_map["csv_file"] = setup_csv_file(filename, header)
        return csv_writer
    else:
        return None


def setup_trajectory_file(filename, no_weights, no_biases, config_map):
    """ Opens the trajectory file if a proper `filename` is given.

    :param filename: filename of trajectory file or None
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_trajectory_file"] = True
        trajectory_writer, config_map["trajectory_file"] = \
            setup_csv_file(filename, ['step', 'loss']
                           + [str("weight")+str(i) for i in range(0,no_weights)]
                           + [str("bias") + str(i) for i in range(0, no_biases)])
        return trajectory_writer
    else:
        return None


def closeFiles(config_map):
    """ Closes the output files if they have been opened.

    :param config_map: configuration dictionary
    """
    if config_map["do_write_csv_file"]:
        assert config_map["csv_file"] is not None
        config_map["csv_file"].close()
        config_map["csv_file"] = None
    if config_map["do_write_trajectory_file"]:
        assert config_map["trajectory_file"] is not None
        config_map["trajectory_file"].close()
        config_map["trajectory_file"] = None
