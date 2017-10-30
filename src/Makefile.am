SUBDIRS = datasets diffusion_maps models samplers

DataDrivenSampler_PYTHON = \
	common.py \
	DataDrivenSampler.py \
	DataOptimizer.py \
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
	DDSAnalyser

CLEANFILES = $(bin_SCRIPTS)
EXTRA_DIST = \
	$(srcdir)/.git-version \
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

$(srcdir)/version.py: $(srcdir)/.git-version
	echo -e "def get_package_version():\n\t return \"$(PACKAGE_NAME) version "`cat $(srcdir)/.git-version`"\";" >$@
