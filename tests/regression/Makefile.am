AUTOM4TE = $(SHELL) $(top_srcdir)/build-aux/missing --run autom4te

TESTSUITE = $(srcdir)/testsuite

DISTCLEANFILES = atconfig

TESTSCRIPTS =

# PLEASE adhere to alphabetical ordering of TESTSCRIPTS
TESTSCRIPTS += \
	$(srcdir)/DataDrivenSampler/testsuite-datadrivensampler.at \
	$(srcdir)/DataDrivenSampler/testsuite-datadrivensampler-simple.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-every_nth.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-friction_constant.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-inverse_temperature.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-step_width.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-no_seeds.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-gla_1st.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-gla_2nd.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-sgld.at \
	$(srcdir)/DataOptimizer/testsuite-dataoptimizer.at \
	$(srcdir)/DataOptimizer/testsuite-dataoptimizer-simple.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-foursquares.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twocircles.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twoclusters.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twospirals.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-step_width.at \
	$(srcdir)/DataOptimizer/Reproducibility/testsuite-dataoptimizer-reproducibility-given_seeds.at \
	$(srcdir)/DataOptimizer/Reproducibility/testsuite-dataoptimizer-reproducibility-no_seeds.at \
	$(srcdir)/TrajectoryAnalyser/testsuite-trajectoryanalyser.at \
	$(srcdir)/TrajectoryAnalyser/testsuite-trajectoryanalyser-simple.at

EXTRA_DIST = \
	testsuite.at \
	$(TESTSUITE) \
	atlocal.in \
	package.m4 \
	ExperimentControl \
	XMLVersionUpdater \
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
