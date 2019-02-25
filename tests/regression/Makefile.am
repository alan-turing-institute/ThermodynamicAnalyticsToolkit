AUTOM4TE = $(SHELL) $(top_srcdir)/build-aux/missing --run autom4te

TESTSUITE = $(srcdir)/testsuite

DISTCLEANFILES = atconfig

TESTSCRIPTS =

# PLEASE adhere to alphabetical ordering of TESTSCRIPTS
TESTSCRIPTS += \
	$(srcdir)/DataDrivenExplorer/testsuite-datadrivenexplorer.at \
	$(srcdir)/DataDrivenExplorer/testsuite-datadrivenexplorer-simple.at \
	$(srcdir)/DataDrivenExplorer/Pruning/testsuite-datadrivenexplorer-pruning.at \
	$(srcdir)/DataDrivenExplorer/SingleTrajectory/testsuite-datadrivenexplorer-single_trajectory.at \
	$(srcdir)/DataDrivenExplorer/MultipleTrajectories/testsuite-datadrivenexplorer-multiple_trajectories.at \
	$(srcdir)/DataDrivenExplorer/MultipleTrajectories/testsuite-datadrivenexplorer-multiple_trajectories-multiple.at \
	$(srcdir)/DataDrivenExplorer/MultipleTrajectories/testsuite-datadrivenexplorer-multiple_trajectories-parallel.at \
	$(srcdir)/DataDrivenExplorer/MultipleTrajectories/testsuite-datadrivenexplorer-multiple_trajectories-single.at \
	$(srcdir)/DataDrivenSampler/testsuite-datadrivensampler.at \
	$(srcdir)/DataDrivenSampler/testsuite-datadrivensampler-simple.at \
	$(srcdir)/DataDrivenSampler/Averages/testsuite-datadrivensampler-averages.at \
	$(srcdir)/DataDrivenSampler/Averages/testsuite-datadrivensampler-averages-burn_in.at \
	$(srcdir)/DataDrivenSampler/FixParameter/testsuite-datadrivensampler-fix_parameter.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-batch_size.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-every_nth.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-every_nth-misspelled.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-friction_constant.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-inverse_temperature.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-progress.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-sql_db.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-step_width.at \
	$(srcdir)/DataDrivenSampler/Options/testsuite-datadrivensampler-options-summaries_path.at \
	$(srcdir)/DataDrivenSampler/Pipelines/testsuite-datadrivensampler-pipelines-csv.at \
	$(srcdir)/DataDrivenSampler/Pipelines/testsuite-datadrivensampler-pipelines-tfrecord.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-lower_boundary.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-lower_upper_boundary.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-lower_upper_boundary-squared.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-samplers.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-tethering.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-tethering-squared.at \
	$(srcdir)/DataDrivenSampler/Prior/testsuite-datadrivensampler-prior-upper_boundary.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-baoab.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-ccadl.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-gla1.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-gla2.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-hmc_1st.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-given_seeds-sgld.at \
	$(srcdir)/DataDrivenSampler/Reproducibility/testsuite-datadrivensampler-reproducibility-no_seeds.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-baoab.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-ccadl.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-gla_1st.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-gla_2nd.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-hmc_1st.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-hmc_2nd.at \
	$(srcdir)/DataDrivenSampler/Samplers/testsuite-datadrivensampler-samplers-sgld.at \
	$(srcdir)/DataDrivenSampler/SavingRestoring/testsuite-datadrivensampler-saving_restoring.at \
	$(srcdir)/DataDrivenSampler/Trajectories/testsuite-datadrivensampler-trajectories.at \
 	$(srcdir)/DataOptimizer/testsuite-dataoptimizer.at \
	$(srcdir)/DataOptimizer/testsuite-dataoptimizer-simple.at \
	$(srcdir)/DataOptimizer/Averages/testsuite-dataoptimizer-averages.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-foursquares.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-mnist.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twocircles.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twoclusters.at \
	$(srcdir)/DataOptimizer/Datasets/testsuite-dataoptimizer-datasets-twospirals.at \
	$(srcdir)/DataOptimizer/Optimizers/testsuite-dataoptimizer-optimizers.at \
	$(srcdir)/DataOptimizer/Optimizers/testsuite-dataoptimizer-optimizers-barzilaiborwein.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-batch_size.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-do_accumulates.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-do_hessians.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-every_nth.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-learning_rate.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-progress.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-sql_db.at \
	$(srcdir)/DataOptimizer/Options/testsuite-dataoptimizer-options-summaries_path.at \
	$(srcdir)/DataOptimizer/Prior/testsuite-dataoptimizer-prior.at \
	$(srcdir)/DataOptimizer/Prior/testsuite-dataoptimizer-prior-lower_upper_boundary.at \
	$(srcdir)/DataOptimizer/Prior/testsuite-dataoptimizer-prior-tethering.at \
	$(srcdir)/DataOptimizer/Reproducibility/testsuite-dataoptimizer-reproducibility-given_seeds.at \
	$(srcdir)/DataOptimizer/Reproducibility/testsuite-dataoptimizer-reproducibility-no_seeds.at \
	$(srcdir)/DataOptimizer/SavingRestoring/testsuite-dataoptimizer-saving_restoring.at \
	$(srcdir)/DatasetParser/testsuite-datasetparser.at \
	$(srcdir)/DatasetParser/testsuite-datasetparser-simple.at \
	$(srcdir)/DatasetParser/Datasets/testsuite-datasetparser-datasets.at \
	$(srcdir)/DatasetParser/Datasets/testsuite-datasetparser-datasets-foursquares.at \
	$(srcdir)/DatasetParser/Datasets/testsuite-datasetparser-datasets-twocircles.at \
	$(srcdir)/DatasetParser/Datasets/testsuite-datasetparser-datasets-twoclusters.at \
	$(srcdir)/DatasetParser/Datasets/testsuite-datasetparser-datasets-twospirals.at \
	$(srcdir)/DatasetWriter/testsuite-datasetwriter.at \
	$(srcdir)/DatasetWriter/testsuite-datasetwriter-simple.at \
	$(srcdir)/DatasetWriter/Datasets/testsuite-datasetwriter-datasets.at \
	$(srcdir)/DatasetWriter/Datasets/testsuite-datasetwriter-datasets-foursquares.at \
	$(srcdir)/DatasetWriter/Datasets/testsuite-datasetwriter-datasets-twocircles.at \
	$(srcdir)/DatasetWriter/Datasets/testsuite-datasetwriter-datasets-twoclusters.at \
	$(srcdir)/DatasetWriter/Datasets/testsuite-datasetwriter-datasets-twospirals.at \
	$(srcdir)/DatasetWriter/Reproducibility/testsuite-datasetwriter-reproducibility-no_seeds.at \
	$(srcdir)/DatasetWriter/Reproducibility/testsuite-datasetwriter-reproducibility-given_seeds.at \
	$(srcdir)/InputSpaceSampler/testsuite-inputspacesampler.at \
	$(srcdir)/InputSpaceSampler/testsuite-inputspacesampler-simple.at \
	$(srcdir)/InputSpaceSampler/testsuite-inputspacesampler-two_clusters_grid.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-exclude_parameters.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-grid.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-grid-directions.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-interval_center_file.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-sgld-trajectory.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-sgld-trajectory-directions.at \
	$(srcdir)/LossFunctionSampler/testsuite-lossfunctionsampler-simple.at \
	$(srcdir)/NumericalDiff/testsuite-numericaldiff.at \
	$(srcdir)/NumericalDiff/testsuite-numericaldiff-simple.at \
	$(srcdir)/NumericalDiff/Comparisons/testsuite-numericaldiff-comparisons.at \
	$(srcdir)/NumericalDiff/Comparisons/testsuite-numericaldiff-comparisons-different_header.at \
	$(srcdir)/NumericalDiff/Comparisons/testsuite-numericaldiff-comparisons-different_length.at \
	$(srcdir)/NumericalDiff/Comparisons/testsuite-numericaldiff-comparisons-different_sign.at \
	$(srcdir)/NumericalDiff/Comparisons/testsuite-numericaldiff-comparisons-different_values.at \
	$(srcdir)/NumericalDiff/Options/testsuite-numericaldiff-options.at \
	$(srcdir)/NumericalDiff/Options/testsuite-numericaldiff-options-column_drop.at \
	$(srcdir)/TrajectoryAnalyser/testsuite-trajectoryanalyser.at \
	$(srcdir)/TrajectoryAnalyser/testsuite-trajectoryanalyser-simple.at \
	$(srcdir)/TrajectoryAnalyser/Averages/testsuite-trajectoryanalyser-averages.at \
	$(srcdir)/TrajectoryAnalyser/Covariance/testsuite-trajectoryanalyser-covariance.at \
	$(srcdir)/TrajectoryAnalyser/Covariance/testsuite-trajectoryanalyser-covariance-gaussian_mixture.at \
	$(srcdir)/TrajectoryAnalyser/Covariance/testsuite-trajectoryanalyser-covariance-gaussian_mixture-walker_individual.at \
	$(srcdir)/TrajectoryAnalyser/DiffusionMap/testsuite-trajectoryanalyser-diffusionmap-eigendecomposition.at \
	$(srcdir)/TrajectoryAnalyser/FreeEnergy/testsuite-trajectoryanalyser-free_energy.at \
	$(srcdir)/TrajectoryAnalyser/FreeEnergy/testsuite-trajectoryanalyser-free_energy-harmonic_oscillator.at \
	$(srcdir)/TrajectoryAnalyser/FreeEnergy/testsuite-trajectoryanalyser-free_energy-histograms.at \
	$(srcdir)/TrajectoryAnalyser/FreeEnergy/testsuite-trajectoryanalyser-free_energy-levelsets.at \
	$(srcdir)/TrajectoryAnalyser/IntegratedAutocorrelationTime/testsuite-trajectoryanalyser-integratedautocorrelationtime.at \
	$(srcdir)/TrajectoryAnalyser/IntegratedAutocorrelationTime/testsuite-trajectoryanalyser-integratedautocorrelationtime-gaussian_mixture.at \
	$(srcdir)/TrajectoryAnalyser/IntegratedAutocorrelationTime/testsuite-trajectoryanalyser-integratedautocorrelationtime-gaussian_mixture-walker_individual.at \
	$(srcdir)/TrajectoryAnalyser/Options/testsuite-trajectoryanalyser-options-drop_burnin-more_steps.at \
	$(srcdir)/TrajectoryAnalyser/Options/testsuite-trajectoryanalyser-options-drop_burnin-single_step.at \
	$(srcdir)/TrajectoryAnalyser/Options/testsuite-trajectoryanalyser-options-every_nth.at

EXTRA_DIST = \
	testsuite.at \
	$(TESTSUITE) \
	atlocal.in \
	package.m4 \
	$(TESTSCRIPTS) \
	DataDrivenExplorer \
	DataDrivenSampler \
	DataOptimizer \
	DatasetParser \
	DatasetWriter \
	InputSpaceSampler \
	LossFunctionSampler \
	NumericalDiff \
	TrajectoryAnalyser

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
