SUBDIRS = datasets diffusion_maps models models/input runtime samplers

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

EXTRA_DIST = \
	$(srcdir)/.git-version

$(srcdir)/version.py: $(srcdir)/.git-version
	echo "def get_package_name():" >$@
	echo "   return \"$(PACKAGE_NAME)\"" >>$@
	echo "def get_package_version():" >>$@
	echo "   return \"$(PACKAGE_VERSION)\"" >>$@
	echo "def get_build_hash():" >>$@
	echo "   return \""`cat $(srcdir)/.git-version`"\"" >>$@
