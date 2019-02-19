SUBDIRS=src tests doc wheel

AUTOMAKE_OPTIONS = subdir-objects

.PHONY: doc wheel
doc:
	mkdir -p ${DX_DOCDIR}
	cd doc/userguide && $(MAKE) doc && cd ../..
	cd doc && $(MAKE) doxygen-doc doxygen-pdf && $(MAKE) install-clean && cd ..

wheel:
	$(MAKE) -C wheel wheel
