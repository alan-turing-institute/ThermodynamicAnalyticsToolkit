SUBDIRS=src tests doc

AUTOMAKE_OPTIONS = subdir-objects

.PHONY: doc
doc:
	mkdir -p ${DX_DOCDIR}
	cd doc && $(MAKE) doxygen-doc

