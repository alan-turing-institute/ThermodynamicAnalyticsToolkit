dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(ThermodynamicAnalyticsToolkit, 0.9.2, frederik.heber@gmail.com)
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

# asciidoc and dblatex
AC_CHECK_PROGS([ASCIIDOC], [asciidoc])
if test -z $ASCIIDOC; then
	AC_MSG_WARN([asciidoc not found, cannot build from asciidoc sources])
fi
AC_CHECK_PROGS([DBLATEX], [dblatex])
if test -z $DBLATEX; then
   AC_MSG_WARN([dblatex not found, cannot render userguide PDF])
fi
AC_CHECK_PROGS([HIGHLIGHT], [source-highlight])
if test -z $HIGHLIGHT; then
   AC_MSG_WARN([GNU's source-highlight not found, cannot syntex-highlight code blocks in userguide])
fi

# We use /bin/true for checking certain tests
# /bin/true is in /usr/in/true on MacOs
AC_CHECK_PROGS([BIN_TRUE], [true])
if test -z $BIN_TRUE; then
	AC_MSG_WARN([Could not find /bin/true or similar."])
	# false is more portable than true: Missing binary will result in 127, which !
	# converts to 0, i.e. always skip
	BIN_TRUE="! /bin/false"
fi

# test for required python modules
AX_PYTHON_MODULE(tensorflow, fatal, python3)
AX_PYTHON_MODULE(pandas, fatal, python3)
AX_PYTHON_MODULE(scipy, fatal, python3)
AX_PYTHON_MODULE(sklearn, fatal, python3)
AX_PYTHON_MODULE(acor, python3)

AC_CONFIG_FILES(
	Makefile
	src/Makefile
	src/TATi/Makefile
	src/TATi/analysis/Makefile
	src/TATi/datasets/Makefile
	src/TATi/diffusion_maps/Makefile
	src/TATi/exploration/Makefile
	src/TATi/models/Makefile
	src/TATi/models/accumulators/Makefile
	src/TATi/models/input/Makefile
	src/TATi/models/networks/Makefile
	src/TATi/models/parameters/Makefile
	src/TATi/models/trajectories/Makefile
	src/TATi/options/Makefile
	src/TATi/runtime/Makefile
	src/TATi/samplers/Makefile
	src/TATi/samplers/dynamics/Makefile
	src/TATi/samplers/grid/Makefile
	src/TATi/tools/Makefile
	tests/Makefile)
AC_CONFIG_TESTDIR([tests/codechecks])
AC_CONFIG_FILES(
	tests/codechecks/atlocal
	tests/codechecks/Makefile)
AC_CONFIG_TESTDIR([tests/integration])
AC_CONFIG_FILES(
	tests/integration/atlocal
	tests/integration/Makefile)
AC_CONFIG_TESTDIR([tests/regression])
AC_CONFIG_FILES(
	tests/regression/atlocal
	tests/regression/Makefile)
AC_CONFIG_TESTDIR([tests/simulation])
AC_CONFIG_FILES(
	tests/simulation/atlocal
	tests/simulation/Makefile)
AC_CONFIG_TESTDIR([tests/tensorflow])
AC_CONFIG_FILES(
	tests/tensorflow/atlocal
	tests/tensorflow/Makefile)
AC_CONFIG_TESTDIR([tests/userguide])
AC_CONFIG_FILES(
	tests/userguide/atlocal
	tests/userguide/Makefile)
AC_CONFIG_FILES([tests/check_package], [chmod +x tests/check_package])
AC_CONFIG_FILES([tests/check_package_version], [chmod +x tests/check_package_version])
AC_CONFIG_FILES([tests/TATiDatasetParser], [chmod +x tests/TATiDatasetParser])
AC_CONFIG_FILES([tests/TATiDatasetWriter], [chmod +x tests/TATiDatasetWriter])
AC_CONFIG_FILES([tests/TATiSampler], [chmod +x tests/TATiSampler])
AC_CONFIG_FILES([tests/TATiExplorer], [chmod +x tests/TATiExplorer])
AC_CONFIG_FILES([tests/TATiOptimizer], [chmod +x tests/TATiOptimizer])
AC_CONFIG_FILES([tests/TATiAnalyser], [chmod +x tests/TATiAnalyser])
AC_CONFIG_FILES([tests/TATiInputSpaceSampler], [chmod +x tests/TATiInputSpaceSampler])
AC_CONFIG_FILES([tests/TATiLossFunctionSampler], [chmod +x tests/TATiLossFunctionSampler])
AC_CONFIG_FILES([tests/NumericalDiff], [chmod +x tests/NumericalDiff])
AC_CONFIG_FILES([tests/run], [chmod +x tests/run])
AC_CONFIG_FILES([
        doc/Makefile
        doc/userguide/Makefile
])
AC_OUTPUT

