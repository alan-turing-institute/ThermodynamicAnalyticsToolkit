# FIXME: this command overrides ${prefix}. If you have run `make install`
# beforehand, you may need to run `make clean` for the wheel to build
# successfully.

.PHONY: wheel

wheel: @PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl

@PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl: setup.py
	$(MAKE) prefix=$(abs_builddir)/build/inst -C .. install-strip && \
	$(PYTHON) setup.py bdist_wheel -d .

wheel-check: @PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl tests/userguide/testsuite.log tests/simulation/testsuite.log

check-logs: wheel-check
	rm -rf _install && $(MKDIR_P) _install && cd _install && \
	unzip ../@PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl && \
	$(MKDIR_P) ../tests && cd ../tests && cp -f $(abs_top_builddir)/tests/run .

tests/simulation/testsuite.log: check-logs
	$(MKDIR_P) tests/simulation && cd tests/simulation && \
	cp -f $(abs_top_builddir)/tests/simulation/atconfig . && \
	cp -f $(abs_top_builddir)/tests/simulation/atlocal . && \
	$(abs_top_srcdir)/tests/simulation/testsuite -k python,!tatisampler AUTOTEST_PATH=$(abs_builddir)/_install

tests/userguide/testsuite.log: check-logs
	$(MKDIR_P) tests/userguide && cd tests/userguide && \
	cp -f $(abs_top_builddir)/tests/userguide/atconfig . && \
	cp -f $(abs_top_builddir)/tests/userguide/atlocal . && \
	$(abs_top_srcdir)/tests/userguide/testsuite -k python,!numericaldiff AUTOTEST_PATH=$(abs_builddir)/_install

clean-local:
	rm -rf build destdir *.whl
	rm -rf _install
	rm -rf tests

EXTRA_DIST = setup.py.in
