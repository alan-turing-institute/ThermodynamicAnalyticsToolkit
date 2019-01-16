TATI_tools_PYTHON = \
	TATiAnalyser.py \
	TATiDatasetParser.py \
	TATiDatasetWriter.py \
	TATiExplorer.py \
	TATiInputSpaceSampler.py \
	TATiLossFunctionSampler.py \
	TATiOptimizer.py \
	TATiSampler.py \
	NumericalDiff.py \
	__init__.py

TATI_toolsdir = $(pythondir)/TATi/tools

bin_SCRIPTS = \
	TATiAnalyser \
	TATiDatasetParser \
	TATiDatasetWriter \
	TATiExplorer \
	TATiInputSpaceSampler \
	TATiLossFunctionSampler \
	TATiOptimizer \
	TATiSampler \
	NumericalDiff

CLEANFILES = $(bin_SCRIPTS)

EXTRA_DIST = \
    run_template.in

# turn run_template.in into a wrapper for the main() functionality inside
# the python script tools
do_substitution = sed -e 's,[@]pythondir[@],$(pythondir),g' \
	-e 's,[@]MODULE[@]'",$@,g" \
	-e 's,[@]PACKAGE[@],$(PACKAGE),g' \
	-e 's,[@]PYTHON[@],$(PYTHON),g' \
	-e 's,[@]VERSION[@],$(VERSION),g'

TATiAnalyser: TATiAnalyser.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiAnalyser
	chmod +x TATiAnalyser

TATiDatasetParser: TATiDatasetParser.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiDatasetParser
	chmod +x TATiDatasetParser

TATiDatasetWriter: TATiDatasetWriter.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiDatasetWriter
	chmod +x TATiDatasetWriter

TATiExplorer: TATiExplorer.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiExplorer
	chmod +x TATiExplorer

TATiInputSpaceSampler: TATiInputSpaceSampler.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiInputSpaceSampler
	chmod +x TATiInputSpaceSampler

TATiLossFunctionSampler: TATiLossFunctionSampler.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiLossFunctionSampler
	chmod +x TATiLossFunctionSampler

TATiOptimizer: TATiOptimizer.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiOptimizer
	chmod +x TATiOptimizer

TATiSampler: TATiSampler.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > TATiSampler
	chmod +x TATiSampler

NumericalDiff: NumericalDiff.py run_template.in Makefile
	$(do_substitution) < $(srcdir)/run_template.in > NumericalDiff
	chmod +x NumericalDiff
