import pandas as pd

class ParsedTrajectory(object):
    """ This class encapsulates a single or multiple trajectories
    parsed from file.

    """
    def __init__(self, filename):
        """

        :param filename: trajectory filename to parse
    *    """
        df_run = pd.read_csv(FLAGS.run_file, sep=',', header=0)
        run = np.asarray(df_run.loc[:, ['step', 'loss', 'kinetic_energy', 'total_energy']])

    if (len(run[:, 0]) > 1) and (FLAGS.drop_burnin >= run[1, 0]):
        if FLAGS.drop_burnin < run[-1, 0]:
            start = next(x[0] for x in enumerate(run[:, 0]) if x[1] > FLAGS.drop_burnin)
        else:
            sys.stderr.write("FLAGS.drop_burnin is too large, no data points left.")
            sys.exit(1)
    else:
        start = 0
    print("Starting run array at " + str(start))

    steps = run[start::FLAGS.every_nth, 0]
    loss = run[start::FLAGS.every_nth, 1]
    kinetic_energy = run[start::FLAGS.every_nth, 2]
    total_energy = run[start::FLAGS.every_nth, 3]

    no_steps = len(steps)

    print("%d steps after dropping burn in." % (no_steps))
    print("%lg average and %lg variance in loss." % (np.average(loss), loss.var()))

    end_list = np.arange(1, FLAGS.steps + 1) * int(no_steps / FLAGS.steps)
    print("Evaluating at steps: " + str(end_list))
