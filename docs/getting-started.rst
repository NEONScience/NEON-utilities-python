Get started with neonutilities
==============================

The neonutilities Python package provides utilities for discovering, downloading,
and working with data files published by the `National Ecological Observatory
Network (NEON) <https://www.neonscience.org/>`_. neonutilities provides functions
for downloading all types of NEON data (tabular, hierarchical, image) and for
joining tabular data files across dates and sites.

Install neonutilities
---------------------

neonutilities has several Python package dependencies including:
``pandas, pyarrow, pyproj, requests``.

From `PyPI <https://pypi.org/project/neonutilities/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installing with ``pip`` or ``uv`` will install dependencies as well; if there is a
problem, use the `requirements file <https://github.com/NEONScience/NEON-utilities-python/blob/main/requirements.txt>`_ in the package documentation.

.. code-block:: shell

   # with pip
   pip install neonutilities

.. code-block:: shell

   # with uv
   uv pip install neonutilities

From `conda-forge <https://github.com/conda-forge/neonutilities-feedstock>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can setup an environment that includes all dependencies (including Python)
with the following command line operations.

With Pixi
~~~~~~~~~

`Pixi <https://pixi.sh/>`_ environments are fully reproducible by default.
From your project directory, initialize a Pixi workspace and then add ``neonutilities``

.. code-block:: shell

   pixi init
   pixi add neonutilities

You can optionally activate an interactive shell with the environment loaded with

.. code-block:: shell

   pixi shell

With conda
~~~~~~~~~~

Create a `conda <https://docs.conda.io/>`_ environment and install ``neonutilities`` from the ``conda-forge`` channel

.. code-block:: shell

   conda create --name neon-env
   conda config --name neon-env --add channels conda-forge
   conda config --name neon-env --remove channels defaults
   conda install --name neon-env neonutilities

and then activate the conda environment

.. code-block:: shell

   conda activate neon-env

From GitHub
^^^^^^^^^^^

We recommend installing from the above package indexes, because the versions of
the package hosted there have been finalized. The development version on GitHub
is likely to be unstable as updates may be in progress.
To install the development version anyway:

.. code-block:: shell

   pip install git+https://github.com/NEONScience/NEON-utilities-python.git@main

Once neonutilities is installed you can import it into Python:

.. code-block:: python

    >>> import neonutilities as nu

For further instructions in using the package, see `Tutorials <https://neon-utilities-python.readthedocs.io/en/latest/tutorials.html>`_.
