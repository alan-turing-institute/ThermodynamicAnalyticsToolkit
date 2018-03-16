DataDrivenSampler_exploration_PYTHON = \
    explorer.py \
    get_executables.py \
    trajectorydata.py \
    trajectorydatacontainer.py \
    trajectoryjob.py \
    trajectoryjob_analyze.py \
    trajectoryjob_check_gradient.py \
    trajectoryjob_extract_minimum_candidates.py \
    trajectoryjob_prune.py \
    trajectoryjob_sample.py \
    trajectoryjob_train.py \
    trajectoryjobqueue.py \
    trajectoryprocess.py \
    trajectoryprocess_sample.py \
    trajectoryprocess_train.py \
    trajectoryprocessqueue.py \
    trajectoryqueue.py \
	__init__.py

DataDrivenSampler_explorationdir = $(pythondir)/DataDrivenSampler/exploration

FORCE:

$(srcdir)/get_executables.py:
	echo "def get_install_path():" >$@
	echo "   return \"$(exec_prefix)/bin/\"" >>$@
