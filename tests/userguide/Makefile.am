AUTOM4TE = $(SHELL) $(top_srcdir)/build-aux/missing --run autom4te

TESTSUITE = $(srcdir)/testsuite

DISTCLEANFILES = atconfig

TESTSCRIPTS =

# PLEASE adhere to alphabetical ordering of TESTSCRIPTS
TESTSCRIPTS += \
	$(srcdir)/cmdline/testsuite-userguide-cmdline.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_averages.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_diffmap.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_free_energy.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-inputspacesampler.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-fix_parameter.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-grid.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-trajectory.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-optimize.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-parse_dataset.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-sample.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-write_dataset.at \
	$(srcdir)/python/testsuite-userguide-python.at \
	$(srcdir)/python/testsuite-userguide-python-analyse.at \
	$(srcdir)/python/testsuite-userguide-python-fix_parameter.at \
	$(srcdir)/python/testsuite-userguide-python-optimize.at \
	$(srcdir)/python/testsuite-userguide-python-optimize_sample.at \
	$(srcdir)/python/testsuite-userguide-python-plot_optimize.at \
	$(srcdir)/python/testsuite-userguide-python-sample.at

EXTRA_DIST = \
	testsuite.at \
	$(TESTSUITE) \
	atlocal.in \
	package.m4 \
	$(TESTSCRIPTS)

max_jobs = 4
 
check-local: atconfig atlocal $(TESTSUITE)
	nrjobs=; \
	for flag in $$MAKEFLAGS; do \
		case $$flag in \
		--* | =*=) ;; \
		*j*) nrjobs="-j$(max_jobs)" ;; \
		esac; \
	done; \
	$(SHELL) '$(TESTSUITE)' $$nrjobs AUTOTEST_PATH='$(abs_top_builddir)/src' $(TESTSUITEFLAGS)

installcheck-local: atconfig atlocal $(TESTSUITE)
	nrjobs=; \
	for flag in $$MAKEFLAGS; do \
		case $$flag in \
		--* | =*=) ;; \
		*j*) nrjobs="-j$(max_jobs)" ;; \
		esac; \
	done; \
	$(SHELL) '$(TESTSUITE)' $$nrjobs AUTOTEST_PATH='$(bindir)' $(TESTSUITEFLAGS)

clean-local:
	test ! -f '$(TESTSUITE)' || \
	$(SHELL) '$(TESTSUITE)' --clean

AUTOTEST = $(AUTOM4TE) --language=autotest
$(TESTSUITE): $(srcdir)/testsuite.at package.m4 $(TESTSCRIPTS)
	$(AUTOTEST) -I '$(srcdir)' -o $@.tmp $@.at
	mv $@.tmp $@

# The `:;' works around a Bash 3.2 bug when the output is not writeable.
package.m4: $(top_srcdir)/configure.ac
	:;{ \
		echo '# Signature of the current package.' && \
		echo 'm4_define([AT_PACKAGE_NAME],      [@PACKAGE_NAME@])' && \
		echo 'm4_define([AT_PACKAGE_TARNAME],   [@PACKAGE_TARNAME@])' && \
		echo 'm4_define([AT_PACKAGE_VERSION],   [@PACKAGE_VERSION@])' && \
		echo 'm4_define([AT_PACKAGE_STRING],    [@PACKAGE_STRING@])' && \
		echo 'm4_define([AT_PACKAGE_BUGREPORT], [@PACKAGE_BUGREPORT@])'; \
	} >'package.m4'
