SUBDIRS = DataDrivenSampler

bin_SCRIPTS = \
	DatasetParser \
	DatasetWriter \
	DDSampler \
	DDSExplorer \
	DDSOptimizer \
	DDSAnalyser \
	InputSpaceSampler \
	LossFunctionSampler \
	NumericalDiff

CLEANFILES = $(bin_SCRIPTS)

EXTRA_DIST = \
	DatasetParser.in \
	DatasetWriter.in \
	DDSampler.in \
	DDSExplorer.in \
	DDSOptimizer.in \
	DDSAnalyser.in \
	InputSpaceSampler.in \
	LossFunctionSampler.in \
	NumericalDiff.in

do_substitution = sed -e 's,[@]pythondir[@],$(pythondir),g' \
	-e 's,[@]PACKAGE[@],$(PACKAGE),g' \
	-e 's,[@]PYTHON[@],$(PYTHON),g' \
	-e 's,[@]VERSION[@],$(VERSION),g' 

DatasetParser: DatasetParser.in Makefile
	$(do_substitution) < $(srcdir)/DatasetParser.in > DatasetParser
	chmod +x DatasetParser

DatasetWriter: DatasetWriter.in Makefile
	$(do_substitution) < $(srcdir)/DatasetWriter.in > DatasetWriter
	chmod +x DatasetWriter

DDSampler: DDSampler.in Makefile
	$(do_substitution) < $(srcdir)/DDSampler.in > DDSampler
	chmod +x DDSampler

DDSExplorer: DDSExplorer.in Makefile
	$(do_substitution) < $(srcdir)/DDSExplorer.in > DDSExplorer
	chmod +x DDSExplorer

DDSOptimizer: DDSOptimizer.in Makefile
	$(do_substitution) < $(srcdir)/DDSOptimizer.in > DDSOptimizer
	chmod +x DDSOptimizer

DDSAnalyser: DDSAnalyser.in Makefile
	$(do_substitution) < $(srcdir)/DDSAnalyser.in > DDSAnalyser
	chmod +x DDSAnalyser

InputSpaceSampler: InputSpaceSampler.in Makefile
	$(do_substitution) < $(srcdir)/InputSpaceSampler.in > InputSpaceSampler
	chmod +x InputSpaceSampler

LossFunctionSampler: LossFunctionSampler.in Makefile
	$(do_substitution) < $(srcdir)/LossFunctionSampler.in > LossFunctionSampler
	chmod +x LossFunctionSampler

NumericalDiff: NumericalDiff.in Makefile
	$(do_substitution) < $(srcdir)/NumericalDiff.in > NumericalDiff
	chmod +x NumericalDiff
