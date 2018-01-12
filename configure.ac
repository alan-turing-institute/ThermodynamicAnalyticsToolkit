dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(DataDrivenSampler, 0.5, frederik.heber@gmail.com)
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_SRCDIR([src/DataDrivenSampler/common.py])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([dist-bzip2 1.5 parallel-tests color-tests tar-pax subdir-objects foreign])
AM_PATH_PYTHON([2.5])

# use doxygen
DX_HTML_FEATURE(ON)
DX_PS_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_INIT_DOXYGEN(DataDrivenSampler, Doxyfile, ${docdir})

# use docbook
AX_CHECK_DOCBOOK

AC_CONFIG_FILES(
	Makefile
	src/Makefile
	src/DataDrivenSampler/Makefile
	src/DataDrivenSampler/datasets/Makefile
	src/DataDrivenSampler/diffusion_maps/Makefile
	src/DataDrivenSampler/exploration/Makefile
	src/DataDrivenSampler/models/Makefile
	src/DataDrivenSampler/models/input/Makefile
	src/DataDrivenSampler/runtime/Makefile
	src/DataDrivenSampler/samplers/Makefile
	tests/Makefile)
AC_CONFIG_TESTDIR([tests/integration])
AC_CONFIG_FILES(
	tests/integration/atlocal
	tests/integration/Makefile)
AC_CONFIG_TESTDIR([tests/regression])
AC_CONFIG_FILES(
	tests/userguide/atlocal
	tests/userguide/Makefile)
AC_CONFIG_TESTDIR([tests/userguide])
AC_CONFIG_FILES(
	tests/regression/atlocal
	tests/regression/Makefile)
AC_CONFIG_FILES([tests/check_pydiffmap], [chmod +x tests/check_pydiffmap])
AC_CONFIG_FILES([tests/DatasetParser], [chmod +x tests/DatasetParser])
AC_CONFIG_FILES([tests/DatasetWriter], [chmod +x tests/DatasetWriter])
AC_CONFIG_FILES([tests/DDSampler], [chmod +x tests/DDSampler])
AC_CONFIG_FILES([tests/DDSOptimizer], [chmod +x tests/DDSOptimizer])
AC_CONFIG_FILES([tests/DDSAnalyser], [chmod +x tests/DDSAnalyser])
AC_CONFIG_FILES([tests/InputSpaceSampler], [chmod +x tests/InputSpaceSampler])
AC_CONFIG_FILES([tests/LossFunctionSampler], [chmod +x tests/LossFunctionSampler])
AC_CONFIG_FILES([tests/NumericalDiff], [chmod +x tests/NumericalDiff])
AC_CONFIG_FILES([tests/userguide/run], [chmod +x tests/userguide/run])
AC_CONFIG_FILES([
        doc/Makefile
        doc/userguide/Makefile
])
AC_CONFIG_FILES([
        doc/userguide/catalog.xml:doc/userguide/catalog.xml.in])

AC_OUTPUT

