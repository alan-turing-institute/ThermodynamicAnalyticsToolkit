# userguide documentation

EXTRA_DIST =

EXTRA_DIST += \
	$(top_builddir)/doc/datadrivensampler.pdf

PYTHON_EXAMPLES = \
	$(srcdir)/userguide/cmdline/analyse_average.sh \
	$(srcdir)/userguide/cmdline/analyse_diffmap.sh \
	$(srcdir)/userguide/cmdline/analyse_free_energy.sh \
	$(srcdir)/userguide/cmdline/exploring.sh \
	$(srcdir)/userguide/cmdline/inputspacesampler.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-fix_parameter.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-grid.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-trajectory.sh \
	$(srcdir)/userguide/cmdline/optimize.sh \
	$(srcdir)/userguide/cmdline/parse_dataset.sh \
	$(srcdir)/userguide/cmdline/sample.sh \
	$(srcdir)/userguide/cmdline/write_dataset.sh \
	$(srcdir)/userguide/python/analyse.py \
	$(srcdir)/userguide/python/explore.py \
	$(srcdir)/userguide/python/fix_parameter.py \
	$(srcdir)/userguide/python/optimize.py \
	$(srcdir)/userguide/python/optimize_sample.py \
	$(srcdir)/userguide/python/plot_optimize.py \
	$(srcdir)/userguide/python/sample.py \
	$(srcdir)/userguide/python/supply_dataset.py \
	$(srcdir)/userguide/python/writing_data.py

# mind: https://stackoverflow.com/questions/39786947/how-to-convert-to-pdf-from-xsl-fo-with-mathml
#
# for MathML to work this need JEuclid and some tricks pulled on Ubuntu 14.04

XSLTPROC=xsltproc
XSLTPROC_FLAGS=@XSLTPROC_FLAGS@ \
	--stringparam paper.type A4 \
	--stringparam glossentry.show.acronym 1 \
	--stringparam glossary.auto.link 1 \
	--xinclude
PDF_STYLESHEET=@DOCBOOK_ROOT@/fo/docbook.xsl
XML_CATALOG_FILES=$(top_builddir)/doc/userguide/catalog.xml
GLOSSARY_FILE=$(top_srcdir)/doc/userguide/glossary.xml
FOP=fop

pdfdir = $(prefix)/share/doc/@PACKAGE@
dist_pdf_DATA = @PACKAGE@.pdf

if have_fop
if have_xsltproc

@PACKAGE@.fo: $(top_srcdir)/doc/userguide/userguide.xml $(top_srcdir)/doc/userguide/bibliography.xml $(top_srcdir)/doc/userguide/glossary.xml $(srcdir)/userguide/version.txt $(GLOSSARY_FILE) $(XML_CATALOG_FILES) $(PYTHON_EXAMPLES)
	XML_CATALOG_FILES=$(XML_CATALOG_FILES) $(XSLTPROC) $(XSLTPROC_FLAGS) -o $@ $(PDF_STYLESHEET) $<
@PACKAGE@.pdf: @PACKAGE@.fo
	$(FOP) -fo $< -pdf $@

else

@PACKAGE@.pdf:
	touch $@

endif
else

@PACKAGE@.pdf:
	touch $@

endif

clean-local:
	$(RM) -f doc/userguide/catalog.xml
	$(RM) -f @PACKAGE@.fo
	$(RM) -f @PACKAGE@.pdf

distclean-local:
	$(RM) -f doc/userguide/catalog.xml
	$(RM) -f @PACKAGE@.fo
	$(RM) -f @PACKAGE@.pdf


SUBDIRS = userguide

EXTRA_DIST += \
	am_doxygen_include.am \
	Doxyfile

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

$(srcdir)/userguide/version.txt: $(srcdir)/.git-version
	cat $(srcdir)/.git-version >$@

include ../src/documentation/Makefile.am

include am_doxygen_include.am
