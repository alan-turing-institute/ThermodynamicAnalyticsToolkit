# FIXME: this command overrides ${prefix}. If you have run `make install`
# beforehand, you may need to run `make clean` for the wheel to build
# successfully.

.PHONY: wheel

wheel: @PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl

@PACKAGE@-@PACKAGE_VERSION@-py3-none-any.whl: setup.py
	$(MAKE) prefix=$(abs_builddir)/build/inst -C .. install-strip && \
	$(PYTHON) setup.py bdist_wheel -d .

clean-local:
	rm -rf build destdir *.whl

EXTRA_DIST = setup.py.in
