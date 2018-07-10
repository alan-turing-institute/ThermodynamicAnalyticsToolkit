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
	$(srcdir)/simulation/complex/optimize.py \
	$(srcdir)/simulation/complex/optimize_sample.py \
	$(srcdir)/simulation/complex/sample.py \
	$(srcdir)/simulation/complex/supply_dataset.py \
	$(srcdir)/simulation/change_dataset_array.py \
	$(srcdir)/simulation/change_dataset_csv.py \
	$(srcdir)/simulation/evaluate_loss.py \
	$(srcdir)/simulation/evaluate_predict.py \
	$(srcdir)/simulation/evaluate_accuracy.py \
	$(srcdir)/simulation/evaluate_gradients.py \
	$(srcdir)/simulation/evaluate_hessians.py \
	$(srcdir)/simulation/requesting_parameters.py \
	$(srcdir)/simulation/setting_parameters.py \
	$(srcdir)/simulation/setting_parameters_walkers.py \
	$(srcdir)/simulation/setting_up_network.py \
	$(srcdir)/simulation/supply_dataset_array.py \
	$(srcdir)/simulation/supply_dataset_csv.py

DEPENDENCIES = \
	bibliography.txt \
	glossary.txt \
	introduction.txt \
	quickstart.txt \
	reference.txt \
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

thermodynamicanalyticstoolkit.xml: $(srcdir)/userguide.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b docbook -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

thermodynamicanalyticstoolkit.html: $(srcdir)/userguide.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b html5 -a toc2 -a numbered -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<
	
thermodynamicanalyticstoolkit.pdf: thermodynamicanalyticstoolkit.xml
	$(DBLATEX) --param=doc.lot.show=figure,table $< $@

all: thermodynamicanalyticstoolkit.pdf thermodynamicanalyticstoolkit.html

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

