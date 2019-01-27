#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###

import argparse
import logging
import pandas as pd
import sys

from TATi.options.commandlineoptions import react_generally_to_options

def parse_parameters():
    parser = argparse.ArgumentParser(
        description="diff two files allowing for slight numerical deviations",
        epilog="Use general_threshold as basis threshold over all columns and\n"+
        "modify single columns using column_threshold."
    )

    # positional arguments
    parser.add_argument('first_file', type=str,
        help='CSV files to compare')
    parser.add_argument('second_file', type=str,
        help='CSV files to compare')

    # please adhere to alphabetical ordering
    parser.add_argument('--column_drop', type=str, nargs='+', default=[],
        help='Names of columns that are not taken into account, i.e. may be contained in one file but not the other')
    parser.add_argument('--column_ignore_sign', type=str, nargs='+', default=[],
        help='Names of columns where flip of sign is ignored, i.e. only absolute values are compared')
    parser.add_argument('--column_threshold', type=str, nargs='+', default=[],
        help='Column-specific threshold to use in triples, e.g. energy 1e-2 relative')
    parser.add_argument('--general_threshold', type=str, nargs='+', default=["1e-8", "absolute"],
        help='General threshold to use for all columns if no other is given, i.e. 1e-2 relative')
    parser.add_argument('--verbose', '-v', action='count',
        help='Level of verbosity during compare')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()

def main(_):
    global FLAGS

    specific_column_names = FLAGS.column_threshold[0::3]
    specific_thresholds = [float(i) for i in FLAGS.column_threshold[1::3]]
    specific_isRelative = [i == "relative" for i in FLAGS.column_threshold[2::3]]

    # parse all files into pandas dataframes
    dataframes = [pd.read_csv(filename, sep=',', header=0) \
                  for filename in [FLAGS.first_file, FLAGS.second_file]]

    # remove drop columns if present
    for name in FLAGS.column_drop:
        for i in range(2):
            if name in dataframes[i].columns:
                logging.info("Dropping column "+name+" from " \
                    +("first" if i==0 else "second")+" file.")
                dataframes[i].pop(name)

    # compare headers
    if dataframes[0].shape[1] != dataframes[1].shape[1]:
        print("There are different number of columns.")
        sys.exit(1)
    if not all([dataframes[0].columns[i] == dataframes[1].columns[i] \
                for i in range(dataframes[0].shape[1])]):
        print("There are already differences in the headers.")
        sys.exit(1)

    # compare lengths
    if dataframes[0].shape[0] != dataframes[1].shape[0]:
        print("The files differ in length.")
        sys.exit(1)

    # gather thresholds per column
    thresholds = []
    isRelative = []
    AbsoluteOnly = []
    for column_name in dataframes[0].columns:
        AbsoluteOnly.append(column_name in FLAGS.column_ignore_sign)
        try:
            i = specific_column_names.index(column_name)
            thresholds.append(specific_thresholds[i])
            isRelative.append(specific_isRelative[i])
        except ValueError:
            thresholds.append(float(FLAGS.general_threshold[0]))
            isRelative.append(FLAGS.general_threshold[1] == "relative")

    # compare the two files using the thresholds
    status = True
    differing_lines = 0
    for col in range(dataframes[0].shape[1]):
        col_name = dataframes[0].columns[col]
        logging.debug("Comparing "+str(dataframes[0][col_name])
                      +" with "+str(dataframes[1][col_name]))
        if AbsoluteOnly[col]:
            error = abs(dataframes[0][col_name]) - abs(dataframes[1][col_name])
        else:
            error = dataframes[0][col_name ] - dataframes[1][ col_name ]
        denominator = [1 if val == 0 else val for val in abs(dataframes[0][ col_name])]
        if isRelative[col]:
            error = error/denominator
        if any(abs(error) > thresholds[col]):
            status = False
            differing_lines = sum(abs(error) > thresholds[col])
            for i in range(len(error)):
                if abs(error[i]) > thresholds[col]:
                    logging.warning(str(i)+","+str(col)+": "+str(dataframes[0][ col_name ][i]) \
                          +" != "+str(dataframes[1][ col_name ][i])+" by "+str(abs(error[i])))

    if status:
        print("Files are equivalent.")
        sys.exit(0)
    else:
        print("Files are NOT equivalent.")
        print(str(differing_lines)+" line(s) differ.")
        sys.exit(1)

def internal_main():
    global FLAGS

    # setup logging
    logging.basicConfig(level=logging.WARNING)

    FLAGS, unparsed = parse_parameters()

    react_generally_to_options(FLAGS, unparsed)

    if (len(FLAGS.general_threshold) != 2):
        print("Exactly two items specify the general threshold.")
        sys.exit(255)

    if (len(FLAGS.column_threshold) % 3 != 0):
        print("Column threshold must be in triples: column name, threshold value, threshold type.")
        sys.exit(255)

    main(None)
