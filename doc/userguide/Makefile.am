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
	$(srcdir)/python/analyse.py \
	$(srcdir)/python/eval_loss_gradient.py \
	$(srcdir)/python/explore.py \
	$(srcdir)/python/fix_parameter.py \
	$(srcdir)/python/general_nn/change_dataset_array.py \
	$(srcdir)/python/general_nn/change_dataset_csv.py \
	$(srcdir)/python/general_nn/evaluate_loss.py \
	$(srcdir)/python/general_nn/evaluate_prediction.py \
	$(srcdir)/python/general_nn/evaluate_accuracy.py \
	$(srcdir)/python/general_nn/evaluate_gradients.py \
	$(srcdir)/python/general_nn/evaluate_hessians.py \
	$(srcdir)/python/general_nn/requesting_parameters.py \
	$(srcdir)/python/general_nn/reset_dataset.py \
	$(srcdir)/python/general_nn/reset_dataset_auto.py \
	$(srcdir)/python/general_nn/setting_parameters.py \
	$(srcdir)/python/general_nn/setting_up_network.py \
	$(srcdir)/python/general_nn/supply_dataset_array.py \
	$(srcdir)/python/general_nn/supply_dataset_csv.py \
	$(srcdir)/python/optimize.py \
	$(srcdir)/python/optimize_sample.py \
	$(srcdir)/python/plot_optimize.py \
	$(srcdir)/python/sample.py \
	$(srcdir)/python/supply_dataset.py \
	$(srcdir)/python/writing_data.py

DEPENDENCIES = \
	bibliography.txt \
	glossary.txt \
	introduction.txt \
	programmersguide.txt \
	programmersguide_computational_graph.txt \
	programmersguide_extensions_samplers.txt \
	programmersguide_neural_networks.txt \
	programmersguide_tensorflow.txt \
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
	$(ASCIIDOC) -b html5 -a toc2 -a icons -a latexmath -a numbered -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

thermodynamicanalyticstoolkit.pdf: thermodynamicanalyticstoolkit.xml
	$(DBLATEX) --param=doc.lot.show=figure,table $< $@

programmersguide.xml: $(srcdir)/programmersguide.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b docbook -a icons -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

programmersguide.html: $(srcdir)/programmersguide.txt $(DEPENDENCIES) $(srcdir)/.git-version
	$(ASCIIDOC) -b html5 -a icons -a latexmath -a toc2 -a numbered -aimagesdir=$(srcdir) -arevdate=$(revdate) -aversion-label=@PACKAGE@ -arevnumber=$(version) -o $@ $<

programmersguide.pdf: programmersguide.xml
	$(DBLATEX) --param=doc.lot.show=figure,table $< $@

all: thermodynamicanalyticstoolkit.pdf thermodynamicanalyticstoolkit.html programmersguide.html programmersguide.pdf

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
