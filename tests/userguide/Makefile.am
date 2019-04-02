AUTOM4TE = $(SHELL) $(top_srcdir)/build-aux/missing --run autom4te

TESTSUITE = $(srcdir)/testsuite

DISTCLEANFILES = atconfig

TESTSCRIPTS =

# PLEASE adhere to alphabetical ordering of TESTSCRIPTS
TESTSCRIPTS += \
	$(srcdir)/cmdline/testsuite-userguide-cmdline.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_averages.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_covariance.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_diffmap.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-analyse_free_energy.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-exploring.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-inputspacesampler.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-exclude_parameters.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-fix_parameter.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-grid.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-lossfunctionsampler-trajectory.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-optimize.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-parse_dataset.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-sample.at \
	$(srcdir)/cmdline/testsuite-userguide-cmdline-write_dataset.at \
	$(srcdir)/python/testsuite-userguide-python.at \
	$(srcdir)/python/testsuite-userguide-python-analyse_averages.at \
	$(srcdir)/python/testsuite-userguide-python-analyse_covariance.at \
	$(srcdir)/python/testsuite-userguide-python-analyse_diffmap.at \
	$(srcdir)/python/testsuite-userguide-python-eval_loss_gradient.at \
	$(srcdir)/python/testsuite-userguide-python-explore.at \
	$(srcdir)/python/testsuite-userguide-python-fix_parameter.at \
	$(srcdir)/python/testsuite-userguide-python-optimize.at \
	$(srcdir)/python/testsuite-userguide-python-optimize_sample.at \
	$(srcdir)/python/testsuite-userguide-python-plot_optimize.at \
	$(srcdir)/python/testsuite-userguide-python-sample.at \
	$(srcdir)/python/testsuite-userguide-python-supply_dataset.at \
	$(srcdir)/python/testsuite-userguide-python-tati_version.at \
	$(srcdir)/python/testsuite-userguide-python-tf_helloworld.at \
	$(srcdir)/python/testsuite-userguide-python-writing_data.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-eval_loss_gradient.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-fix_parameter.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-gla2_sampler.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-momenta_invalid_sampler.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-optimize.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-optimize_sample.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-sample.at \
	$(srcdir)/simulation/complex/testsuite-userguide-simulation-complex-supply_dataset.at \
	$(srcdir)/simulation/examples/testsuite-userguide-simulation-examples.at \
	$(srcdir)/simulation/examples/testsuite-userguide-simulation-examples-harmonic_oscillator_dataset.at \
	$(srcdir)/simulation/examples/testsuite-userguide-simulation-examples-harmonic_oscillator_sampling.at \
	$(srcdir)/simulation/testsuite-userguide-simulation.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-change_dataset_array.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-change_dataset_csv.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_accuracy.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_dataset.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_gradients.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_hessians.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_loss.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-evaluate_predict.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-fit.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-fix_parameter.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-get_activations.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-get_losses.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-get_options.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-requesting_momenta.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-requesting_parameters.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-sample.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-setting_momenta.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-setting_momenta_walkers.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-setting_parameters.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-setting_parameters_walkers.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-setting_up_network.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-supply_dataset_array.at \
	$(srcdir)/simulation/testsuite-userguide-simulation-supply_dataset_csv.at


EXTRA_DIST = \
	testsuite.at \
	$(TESTSUITE) \
	atlocal.in \
	cmdline \
	package.m4 \
	python \
	simulation \
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
