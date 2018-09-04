PYTHON_EXAMPLES = \
	$(srcdir)/cmdline/analyse_average.sh \
	$(srcdir)/cmdline/analyse_diffmap.sh \
	$(srcdir)/cmdline/analyse_free_energy.sh \
	$(srcdir)/cmdline/exploring.sh \
	$(srcdir)/cmdline/inputspacesampler.sh \
	$(srcdir)/cmdline/lossfunctionsampler-exclude_parameters.sh \
	$(srcdir)/cmdline/lossfunctionsampler-fix_parameter.sh \
	$(srcdir)/cmdline/lossfunctionsampler-grid.sh \
	$(srcdir)/cmdline/lossfunctionsampler-trajectory.sh \
	$(srcdir)/cmdline/optimize.sh \
	$(srcdir)/cmdline/parse_dataset.sh \
	$(srcdir)/cmdline/sample.sh \
	$(srcdir)/cmdline/write_dataset.sh \
	$(srcdir)/python/analyse_averages.py \
	$(srcdir)/python/analyse_diffmap.py \
	$(srcdir)/python/eval_loss_gradient.py \
	$(srcdir)/python/explore.py \
	$(srcdir)/python/fix_parameter.py \
	$(srcdir)/python/optimize.py \
	$(srcdir)/python/optimize_sample.py \
	$(srcdir)/python/plot_optimize.py \
	$(srcdir)/python/sample.py \
	$(srcdir)/python/supply_dataset.py \
	$(srcdir)/python/writing_data.py \
	$(srcdir)/simulation/complex/eval_loss_gradient.py \
	$(srcdir)/simulation/complex/fix_parameter.py \
	$(srcdir)/simulation/complex/gla2_sampler.py \
	$(srcdir)/simulation/complex/momenta_invalid_sampler.py \
	$(srcdir)/simulation/complex/optimize.py \
	$(srcdir)/simulation/complex/optimize_sample.py \
	$(srcdir)/simulation/complex/sample.py \
	$(srcdir)/simulation/complex/supply_dataset.py \
	$(srcdir)/simulation/examples/harmonic_oscillator_dataset.py \
	$(srcdir)/simulation/examples/harmonic_oscillator_sampling.py \
	$(srcdir)/simulation/change_dataset_array.py \
	$(srcdir)/simulation/change_dataset_csv.py \
	$(srcdir)/simulation/evaluate_accuracy.py \
	$(srcdir)/simulation/evaluate_dataset.py \
	$(srcdir)/simulation/evaluate_gradients.py \
	$(srcdir)/simulation/evaluate_hessians.py \
	$(srcdir)/simulation/evaluate_loss.py \
	$(srcdir)/simulation/evaluate_predict.py \
	$(srcdir)/simulation/fit.py \
	$(srcdir)/simulation/initializing_momenta.py \
	$(srcdir)/simulation/requesting_parameters.py \
	$(srcdir)/simulation/requesting_momenta.py \
	$(srcdir)/simulation/sample.py \
	$(srcdir)/simulation/setting_momenta.py \
	$(srcdir)/simulation/setting_momenta_walkers.py \
	$(srcdir)/simulation/setting_parameters.py \
	$(srcdir)/simulation/setting_parameters_walkers.py \
	$(srcdir)/simulation/setting_up_network.py \
	$(srcdir)/simulation/supply_dataset_array.py \
	$(srcdir)/simulation/supply_dataset_csv.py

DEPENDENCIES = \
	bibliography.txt \
	glossary.txt \
	introduction.txt \
	programmersguide.txt \
	programmersguide_computational_graph.txt \
    programmersguide_debugging_tips.txt \
	programmersguide_extensions_samplers.txt \
	programmersguide_neural_networks.txt \
	programmersguide_tati_buildsystem.txt \
	programmersguide_tati_structure.txt \
	programmersguide_tati_versioncontrol.txt \
	programmersguide_tensorflow.txt \
	quickstart_cmdline.txt \
	quickstart_sampling.txt \
	quickstart_simulation.txt \
	quickstart.txt \
	reference.txt \
	reference_examples.txt \
	reference_examples_harmonic_oscillator.txt \
	reference_implementing_sampler.txt \
	reference_samplers.txt \
	reference_simulation.txt \
	roadmap.txt \
	tensorflow_flaws.txt \
	userguide.txt \
	$(PYTHON_EXAMPLES)

EXTRA_DIST = \
	cmdline \
	pictures \
	python \
	$(DEPENDENCIES)

versionfile = $(srcdir)/.git-version
version = $(shell cat $(versionfile))
revdate = $(shell date +%Y-%m-%d)

#thermodynamicanalyticstoolkit.xml: $(srcdir)/userguide.txt

#thermodynamicanalyticstoolkit.html: $(srcdir)/userguide.txt

%.xml: $(srcdir)/%.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b docbook -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

%.html: $(srcdir)/%.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b html5 -a toc2 -a latexmath -a numbered -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

%.pdf: %.xml
	$(DBLATEX) --param=doc.lot.show=figure,table $< $@

all: programmersguide.pdf programmersguide.html \
	userguide.pdf userguide.html \
    roadmap.pdf roadmap.html

FORCE:

$(srcdir)/.git-version: FORCE
	@if (test -d $(top_srcdir)/.git && cd $(srcdir) \
		&& { git describe --dirty --long || git describe --long; } ) > .git-version-t 2>/dev/null \
		&& ! diff .git-version-t $(srcdir)/.git-version >/dev/null 2>&1; then \
		mv -f .git-version-t $(srcdir)/.git-version; \
	else \
		rm -f .git-version-t; \
	fi

EXTRA_DIST += \
	$(srcdir)/.git-version

$(srcdir)/version.txt: $(srcdir)/.git-version
	cat $(srcdir)/.git-version >$@
