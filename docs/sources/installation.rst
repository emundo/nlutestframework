Installation
============

.. Note:: Consider installing the framework in a `virtualenv <https://virtualenv.pypa.io/>`_ instead of globally. Just run ``make env/`` in the project root to create the env and activate it afterwards using ``source env/bin/activate``.

.. Warning:: You might need the dev headers for your Python version to build and install the framework. These dev headers should be available via your os' package manager, under a name similar to `libpython3.8-dev`.

Stable versions of this framework can be installed via `pip <https://pip.pypa.io/>`_, the package is called ``nlutestframework``.

Building from source is just as easy. Make sure you have a recent version of `setuptools <https://setuptools.readthedocs.io/>`_ installed, then run ``python setup.py install``.

Building the documentation
--------------------------

To build the documentation, change into the ``docs`` directory, run ``pip install -r requirements.txt`` to install the documentation-specific dependencies, followed by ``make html`` (or ``make.bat html``, if you're on Windows) to build the HTML documentation.

Running the tests
-----------------

To run the tests, install `pytest <https://docs.pytest.org/>`_ (available on pip) and run ``pytest`` from the project root. If you have used the ``make env/`` command to build a virtual environment, you can just run ``make test`` to execute the tests inside of the environment.
