SUBDIRS = datasets diffusion_maps models runtime samplers

DataDrivenSampler_PYTHON = \
	common.py \
	TrajectoryAnalyser.py \
	$(srcdir)/version.py \
	__init__.py

DataDrivenSamplerdir = $(pythondir)/DataDrivenSampler

FORCE:
$(srcdir)/.git-version: FORCE
	@if (test -d $(top_srcdir)/.git && cd $(srcdir) \
		&& { git describe --dirty --always || git describe; } ) > .git-version-t 2>/dev/null \
		&& ! diff .git-version-t $(srcdir)/.git-version >/dev/null 2>&1; then \
		mv -f .git-version-t $(srcdir)/.git-version; \
	else \
		rm -f .git-version-t; \
	fi

bin_SCRIPTS = \
	DDSampler \
	DDSOptimizer \
	DDSAnalyser \
	LossFunctionSampler

CLEANFILES = $(bin_SCRIPTS)
EXTRA_DIST = \
	$(srcdir)/.git-version \
	DDSampler.in \
	DDSOptimizer.in \
	DDSAnalyser.in \
	LossFunctionSampler.in

do_substitution = sed -e 's,[@]pythondir[@],$(pythondir),g' \
	-e 's,[@]PACKAGE[@],$(PACKAGE),g' \
	-e 's,[@]PYTHON[@],$(PYTHON),g' \
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

LossFunctionSampler: LossFunctionSampler.in Makefile
	$(do_substitution) < $(srcdir)/LossFunctionSampler.in > LossFunctionSampler
	chmod +x LossFunctionSampler

$(srcdir)/version.py: $(srcdir)/.git-version
	echo -e "def get_package_name():\n\t return \"$(PACKAGE_NAME)\"" >$@
	echo -e "def get_package_version():\n\t return \"$(PACKAGE_VERSION)\"" >>$@
	echo -e "def get_build_hash():\n\t return \""`cat $(srcdir)/.git-version`"\"" >>$@
