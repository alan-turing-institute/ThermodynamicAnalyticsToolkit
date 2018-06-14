dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(ThermodynamicAnalyticsToolkit, 0.7, frederik.heber@gmail.com)
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_SRCDIR([src/TATi/common.py])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([dist-bzip2 1.5 parallel-tests color-tests tar-pax subdir-objects])
AM_PATH_PYTHON([2.5])

# use doxygen
DX_HTML_FEATURE(ON)
DX_PS_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_INIT_DOXYGEN(ThermodynamicAnalyticsToolkit, Doxyfile, ${docdir})

# use docbook
AX_CHECK_DOCBOOK

AC_CONFIG_FILES(
	Makefile
	src/Makefile
	src/TATi/Makefile
	src/TATi/datasets/Makefile
	src/TATi/diffusion_maps/Makefile
	src/TATi/exploration/Makefile
	src/TATi/models/Makefile
	src/TATi/models/input/Makefile
	src/TATi/runtime/Makefile
	src/TATi/samplers/Makefile
	tests/Makefile)
AC_CONFIG_TESTDIR([tests/integration])
AC_CONFIG_FILES(
	tests/integration/atlocal
	tests/integration/Makefile)
AC_CONFIG_TESTDIR([tests/regression])
AC_CONFIG_FILES(
	tests/regression/atlocal
	tests/regression/Makefile)
AC_CONFIG_FILES(
	tests/tensorflow/atlocal
	tests/tensorflow/Makefile)
AC_CONFIG_TESTDIR([tests/tensorflow])
AC_CONFIG_FILES(
	tests/userguide/atlocal
	tests/userguide/Makefile)
AC_CONFIG_TESTDIR([tests/userguide])
AC_CONFIG_FILES([tests/check_pydiffmap], [chmod +x tests/check_pydiffmap])
AC_CONFIG_FILES([tests/TATiDatasetParser], [chmod +x tests/TATiDatasetParser])
AC_CONFIG_FILES([tests/TATiDatasetWriter], [chmod +x tests/TATiDatasetWriter])
AC_CONFIG_FILES([tests/TATiSampler], [chmod +x tests/TATiSampler])
AC_CONFIG_FILES([tests/TATiExplorer], [chmod +x tests/TATiExplorer])
AC_CONFIG_FILES([tests/TATiOptimizer], [chmod +x tests/TATiOptimizer])
AC_CONFIG_FILES([tests/TATiAnalyser], [chmod +x tests/TATiAnalyser])
AC_CONFIG_FILES([tests/TATiInputSpaceSampler], [chmod +x tests/TATiInputSpaceSampler])
AC_CONFIG_FILES([tests/TATiLossFunctionSampler], [chmod +x tests/TATiLossFunctionSampler])
AC_CONFIG_FILES([tests/NumericalDiff], [chmod +x tests/NumericalDiff])
AC_CONFIG_FILES([tests/userguide/run], [chmod +x tests/userguide/run])
AC_CONFIG_FILES([tests/tensorflow/run], [chmod +x tests/tensorflow/run])
AC_CONFIG_FILES([
        doc/Makefile
        doc/userguide/Makefile
])
AC_CONFIG_FILES([
        doc/userguide/catalog.xml:doc/userguide/catalog.xml.in])

AC_OUTPUT

