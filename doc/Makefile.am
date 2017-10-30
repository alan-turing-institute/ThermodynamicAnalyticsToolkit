# userguide documentation

EXTRA_DIST =

EXTRA_DIST += \
	$(top_builddir)/doc/datadrivensampler.pdf

XSLTPROC=xsltproc
XSLTPROC_FLAGS=@XSLTPROC_FLAGS@ \
	--stringparam paper.type A4
PDF_STYLESHEET=@DOCBOOK_ROOT@/fo/docbook.xsl
XML_CATALOG_FILES=$(top_builddir)/doc/userguide/catalog.xml
FOP=fop

pdfdir = $(prefix)/share/doc/@PACKAGE@
dist_pdf_DATA = @PACKAGE@.pdf

if have_fop
if have_xsltproc

@PACKAGE@.fo: $(top_srcdir)/doc/userguide/userguide.xml
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
