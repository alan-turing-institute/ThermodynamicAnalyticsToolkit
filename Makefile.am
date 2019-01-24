SUBDIRS=src tests doc

AUTOMAKE_OPTIONS = subdir-objects

.PHONY: doc
doc:
	mkdir -p ${DX_DOCDIR}
	cd doc/userguide && $(MAKE) doc && cd ../..
	cd doc && $(MAKE) doxygen-doc doxygen-pdf && $(MAKE) install-clean && cd ..

