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

 * tensorflow
 * numpy
 * pandas
 * scipy
 * scikit-learn

It has received financial support from a seed funding grant and through a 
Rutherford fellowship from the Alan Turing Institute in London (R-SIS-003, 
R-RUT-001) and EPSRC grant no. EP/P006175/1 (Data Driven Coarse Graining using
Space-Time Diffusion Maps, B. Leimkuhler PI)

Please refer to INSTALL for installation instructions. When cloning from
github please call the bootstrap.sh script (requiring installed autotools and
automake packages). 

Furthermore, for installation from a cloned git repository or a pure source
tarball, the following non-standard packages are recommended for creating the
userguide:

 * xsltproc, fop, docbook-xsl

After installation (configure, make, make install) there is a userguide 
"thermodynamicanalyticstoolkit.pdf" for full reference to end users.
Note that there is also a documentation aimed at programmers based on doxygen.
