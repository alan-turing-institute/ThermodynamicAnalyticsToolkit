Thermodynamic Analytics ToolkIt (TATi)
======================================

Thermodynamic Analytics Toolkit is a sampling-based approach to understand the
effectiveness of neural networks training and investigate their loss manifolds.

It uses Tensorflow (https://www.tensorflow.org/) as neural network
framework and implements advanced sampling algorithms on top of it. It contains
both a rapid prototyping platform for new sampling methods and also an analysis
framework to understand the intricacies of the loss manifold in terms of
averages, covariance, diffusion maps, and free energy.

Please take a look at the extensive [userguide](https://alan-turing-institute.github.io/ThermodynamicAnalyticsToolkit/).

Dependencies
------------

In total, we depend on the following python packages:

 * tensorflow (1.4.1, 1.6-1.10; 1.5 is not recommended)
 * numpy
 * pandas
 * scipy
 * scikit-learn
 * acor (see the userguide for installation instructions)

Furthermore, for installation from a cloned git repository or a pure source
tarball, the following non-python packages are required for creating all 
userguides,

 * doxygen,
 * asciidoc, dblatex
 * pdflatex,

and for running all tests,

 * awk, sqlite3.

Finally, there are some optional python packages:

 * pydiffmap: allows diffusion map analysis through pydiffmap package
 * tqdm: allows displaying progress bar during training and sampling

Installation
------------

Use one of the following ways:

- `pip install tati`
- Grab latest [release](https://github.com/alan-turing-institute/ThermodynamicAnalyticsToolkit/releases), extract and `configure --prefix=<your choice>`, `make`, `make install`
- `git clone https://github.com/alan-turing-institute/ThermodynamicAnalyticsToolkit.git` and `configure --prefix=<your choice>`, `make`, `make install`.

For more information please refer to the userguide (see the 
[releases](https://github.com/alan-turing-institute/ThermodynamicAnalyticsToolkit/releases) on github or [as html version](https://alan-turing-institute.github.io/ThermodynamicAnalyticsToolkit/)) 
for installation instructions.

Alternatively, the userguide PDF is also contained in the release tarballs in 
folder **doc/userguide**.
As a fall-back the asciidoc userguide files reside in **doc/userguide** 
and are perfectly human-readable, see **doc/userguide/introduction.txt**.
As a last fall-back have a look at INSTALL for general instructions on how to
installing a package maintained by autotools, automake.

When cloning from github please call the `./bootstrap.sh` script (requiring
installed autotools and automake packages).

NOTE: If you only want to *use* the package and *do not plan to submit code*, 
it is strongly advised to *use the PyPI package (using `pip`) or the "release" 
tarballs* instead of cloning the repository directly.

Documentation
-------------

In general, the documentation is maintained in the folder **doc**. The asciidoc
userguide files reside in **doc/userguide** and are human-readable in your 
preferred editor if every other option fails.

There are multiple guides to help you:

- Userguide: user manual on how to install and use TATi
- Programmer's guide: manual on basic programming with Tensorflow and TATi
- API reference: doxygen-generated API reference

After installation (configure, make, make doc, make install) these guides
can be found in the typical documentation directory (e.g., 
**share/doc/thermodynamicanalyticstoolkit/** depending on your OS).

Note that all of the above guides are also available as *html* versions after
installation.

Acknowledgments
---------------

TATi has received financial support from a seed funding grant and through a 
Rutherford fellowship from the Alan Turing Institute in London (R-SIS-003, 
R-RUT-001), from an EPSRC grant no. EP/P006175/1 (Data Driven Coarse Graining
using Space-Time Diffusion Maps, B. Leimkuhler PI), and also from a Microsoft
Azure  Sponsorship (MS-AZR-0143P).

