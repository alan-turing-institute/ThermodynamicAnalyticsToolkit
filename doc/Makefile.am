# userguide documentation

EXTRA_DIST =

do_substitution = sed -e 's,[@]pythonbindir[@],$(PYTHON),g'

py_filter: py_filter.in
	$(do_substitution) < $(srcdir)/py_filter.in > py_filter
	chmod +x py_filter

all: py_filter

precusor-doxygen-doc: py_filter

clean-local:
	rm $(DX_CLEANFILES)

distclean-local:

SUBDIRS = userguide

EXTRA_DIST += \
	am_doxygen_include.am \
	Doxyfile \
	py_filter.in \
	@PACKAGE@-API-reference.pdf

include ../src/documentation/Makefile.am

include am_doxygen_include.am

dist-local: $(DX_PDF_GOAL) $(DX_HTML_GOAL)
