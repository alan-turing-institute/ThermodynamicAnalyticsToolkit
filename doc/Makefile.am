# userguide documentation

EXTRA_DIST =

do_substitution = sed -e 's,[@]pythonbindir[@],$(PYTHON),g'

py_filter: py_filter.in
	$(do_substitution) < $(srcdir)/py_filter.in > py_filter
	chmod +x py_filter

all: py_filter

precusor-doxygen-doc: py_filter

clean-local:

distclean-local:

SUBDIRS = userguide

EXTRA_DIST += \
	am_doxygen_include.am \
	Doxyfile \
	py_filter.in

include ../src/documentation/Makefile.am

include am_doxygen_include.am
