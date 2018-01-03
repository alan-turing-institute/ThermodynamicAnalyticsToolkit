# userguide documentation

EXTRA_DIST =

EXTRA_DIST += \
	$(top_builddir)/doc/datadrivensampler.pdf

PYTHON_EXAMPLES = \
	$(srcdir)/userguide/cmdline/analyse_average.sh \
	$(srcdir)/userguide/cmdline/analyse_diffmap.sh \
	$(srcdir)/userguide/cmdline/analyse_free_energy.sh \
	$(srcdir)/userguide/cmdline/inputspacesampler.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-fix_parameter.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-grid.sh \
	$(srcdir)/userguide/cmdline/lossfunctionsampler-trajectory.sh \
	$(srcdir)/userguide/cmdline/optimize.sh \
	$(srcdir)/userguide/cmdline/parse_dataset.sh \
	$(srcdir)/userguide/cmdline/sample.sh \
	$(srcdir)/userguide/cmdline/write_dataset.sh \
	$(srcdir)/userguide/python/analyse.py \
	$(srcdir)/userguide/python/optimize.py \
	$(srcdir)/userguide/python/optimize_sample.py \
	$(srcdir)/userguide/python/plot_optimize.py \
	$(srcdir)/userguide/python/sample.py \
	$(srcdir)/userguide/python/writing_data.py

# mind: https://stackoverflow.com/questions/39786947/how-to-convert-to-pdf-from-xsl-fo-with-mathml
#
# for MathML to work this need JEuclid and some tricks pulled on Ubuntu 14.04

XSLTPROC=xsltproc
XSLTPROC_FLAGS=@XSLTPROC_FLAGS@ \
	--stringparam paper.type A4 \
	--xinclude
PDF_STYLESHEET=@DOCBOOK_ROOT@/fo/docbook.xsl
XML_CATALOG_FILES=$(top_builddir)/doc/userguide/catalog.xml
FOP=fop

pdfdir = $(prefix)/share/doc/@PACKAGE@
dist_pdf_DATA = @PACKAGE@.pdf

if have_fop
if have_xsltproc

@PACKAGE@.fo: $(top_srcdir)/doc/userguide/userguide.xml $(XML_CATALOG_FILES) $(PYTHON_EXAMPLES)
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

include ../src/documentation/Makefile.am

include am_doxygen_include.am
