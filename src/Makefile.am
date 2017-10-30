SUBDIRS = datasets diffusion_maps models samplers

DataDrivenSampler_PYTHON = \
	common.py \
	DataDrivenSampler.py \
	DataOptimizer.py \
	TrajectoryAnalyser.py \
	__init__.py

DataDrivenSamplerdir = $(pythondir)/DataDrivenSampler

bin_SCRIPTS = \
	DDSampler \
	DDSOptimizer \
	DDSAnalyser

CLEANFILES = $(bin_SCRIPTS)
EXTRA_DIST = \
	DDSampler.in \
	DDSOptimizer.in \
	DDSAnalyser.in

do_substitution = sed -e 's,[@]pythondir[@],$(pythondir),g' \
	-e 's,[@]PACKAGE[@],$(PACKAGE),g' \
	-e 's,[@]VERSION[@],$(VERSION),g' 

DDSampler: DDSampler.in Makefile
	$(do_substitution) < $(srcdir)/DDSampler.in > DDSampler
	chmod +x DDSampler

DDSOptimizer: DDSOptimizer.in Makefile
	$(do_substitution) < $(srcdir)/DDSOptimizer.in > DDSOptimizer
	chmod +x DDSOptimizer

DDSAnalyser: DDSAnalyser.in Makefile
	$(do_substitution) < $(srcdir)/DDSAnalyser.in > DDSAnalyser
	chmod +x DDSAnalyser
