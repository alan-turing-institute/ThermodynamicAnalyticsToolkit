nodist_TATI_exploration_PYTHON = \
    $(builddir)/get_executables.py

TATI_exploration_PYTHON = \
    explorer.py \
    trajectorydata.py \
    trajectorydatacontainer.py \
    trajectoryjob.py \
    trajectoryjob_analyze.py \
    trajectoryjob_check_gradient.py \
    trajectoryjob_extract_minimum_candidates.py \
    trajectoryjob_prune.py \
    trajectoryjob_sample.py \
    trajectoryjob_train.py \
    trajectoryjobid.py \
    trajectoryjobqueue.py \
    trajectoryprocess.py \
    trajectoryprocess_sample.py \
    trajectoryprocess_train.py \
    trajectoryprocessqueue.py \
    trajectoryqueue.py \
	__init__.py

TATI_explorationdir = $(pythondir)/TATi/exploration

FORCE:

$(builddir)/get_executables.py:
	echo "def get_install_path():" >$@
	echo "   return \"$(exec_prefix)/bin/\"" >>$@
