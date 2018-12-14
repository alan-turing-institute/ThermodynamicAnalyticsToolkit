Thermodynamic Analytics Toolkit
===============================

Thermodynamic Analytics Toolkit is a sampling-based approach to understand the
effectiveness of neural networks training.

It uses Tensorflow (https://www.tensorflow.org/) as neural network
framework and implements sampling algorithms on top of it. These
samplers create trajectories with certain statistical properties
that are used to extract quantities such as slow reaction coordinates
and free energy.

In total, we depend on the following python packages:

 * tensorflow (1.4.1, 1.6-1.10; 1.5 is not recommended)
 * numpy
 * pandas
 * scipy
 * scikit-learn

It has received financial support from a seed funding grant and through a 
Rutherford fellowship from the Alan Turing Institute in London (R-SIS-003, 
R-RUT-001) and EPSRC grant no. EP/P006175/1 (Data Driven Coarse Graining using
Space-Time Diffusion Maps, B. Leimkuhler PI)

In general, see doc/userguide for all manuals and guides accompanying this
package.

Please refer to the guide (see introduction.txt) for installation instructions.
As a fall-back have a look at INSTALL. When cloning from github please call 
the bootstrap.sh script (requiring installed autotools and automake packages).

NOTE: If you only want to *use* the package and *do not plan to submit code*, 
it is strongly advised to *use the "release" tarballs* instead of cloning the 
repository directly.

Furthermore, for installation from a cloned git repository or a pure source
tarball, the following non-standard packages are required for creating the
userguide and running all tests:

 * asciidoc, dblatex
 * gawk, sqlite3

After installation (configure, make, make install) there is a userguide 
"thermodynamicanalyticstoolkit.pdf" for full reference to end users. 
Alternatively, this userguide is always contained in the release tarballs.
Note that there is also a documentation aimed at programmers based on doxygen
(make doc).

Finally, there are some optional packages:

 * pydiffmap: allows diffusion map analysis through pydiffmap package
 * tqdm: allows displaying progress bar during training and sampling

