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
	TATiAnalyser.in \
	TATiDatasetParser.in \
	TATiDatasetWriter.in \
	TATiExplorer.in \
	TATiInputSpaceSampler.in \
	TATiLossFunctionSampler.in \
	TATiOptimizer.in \
	TATiSampler.in \
	NumericalDiff.in

do_substitution = sed -e 's,[@]pythondir[@],$(pythondir),g' \
	-e 's,[@]PACKAGE[@],$(PACKAGE),g' \
	-e 's,[@]PYTHON[@],$(PYTHON),g' \
	-e 's,[@]VERSION[@],$(VERSION),g'

TATiAnalyser: TATiAnalyser.in Makefile
	$(do_substitution) < $(srcdir)/TATiAnalyser.in > TATiAnalyser
	chmod +x TATiAnalyser

TATiDatasetParser: TATiDatasetParser.in Makefile
	$(do_substitution) < $(srcdir)/TATiDatasetParser.in > TATiDatasetParser
	chmod +x TATiDatasetParser

TATiDatasetWriter: TATiDatasetWriter.in Makefile
	$(do_substitution) < $(srcdir)/TATiDatasetWriter.in > TATiDatasetWriter
	chmod +x TATiDatasetWriter

TATiExplorer: TATiExplorer.in Makefile
	$(do_substitution) < $(srcdir)/TATiExplorer.in > TATiExplorer
	chmod +x TATiExplorer

TATiInputSpaceSampler: TATiInputSpaceSampler.in Makefile
	$(do_substitution) < $(srcdir)/TATiInputSpaceSampler.in > TATiInputSpaceSampler
	chmod +x TATiInputSpaceSampler

TATiLossFunctionSampler: TATiLossFunctionSampler.in Makefile
	$(do_substitution) < $(srcdir)/TATiLossFunctionSampler.in > TATiLossFunctionSampler
	chmod +x TATiLossFunctionSampler

TATiOptimizer: TATiOptimizer.in Makefile
	$(do_substitution) < $(srcdir)/TATiOptimizer.in > TATiOptimizer
	chmod +x TATiOptimizer

TATiSampler: TATiSampler.in Makefile
	$(do_substitution) < $(srcdir)/TATiSampler.in > TATiSampler
	chmod +x TATiSampler

NumericalDiff: NumericalDiff.in Makefile
	$(do_substitution) < $(srcdir)/NumericalDiff.in > NumericalDiff
	chmod +x NumericalDiff
