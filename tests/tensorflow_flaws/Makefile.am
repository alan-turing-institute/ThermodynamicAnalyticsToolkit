AUTOM4TE = $(SHELL) $(top_srcdir)/build-aux/missing --run autom4te

TESTSUITE = $(srcdir)/testsuite

DISTCLEANFILES = atconfig

TESTSCRIPTS =

# PLEASE adhere to alphabetical ordering of TESTSCRIPTS
TESTSCRIPTS += \
	$(srcdir)/testsuite.at \
	$(srcdir)/NonVariables/testsuite-nonvariables.at \
	$(srcdir)/NonVariables/testsuite-nonvariables-concat_not_variable.at \
	$(srcdir)/Reproducibility/testsuite-reproducibility.at \
	$(srcdir)/Reproducibility/testsuite-reproducibility-global_seed.at \
	$(srcdir)/Reproducibility/testsuite-reproducibility-repeat_shuffle_reproducibility.at

EXTRA_DIST = \
	testsuite.at \
	$(TESTSUITE) \
	atlocal.in \
	package.m4 \
	$(TESTSCRIPTS) \
	NonVariables \
	Reproducibility

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
