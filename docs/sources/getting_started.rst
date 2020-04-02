Getting Started
===============

**Step 1**: :doc:`Install the framework <installation>`.

**Step 2**: Prepare the data sets and data set loaders you want to benchmark on. This repository comes with a few data sets that you can use out-of-the-box. You can find them in the ``data`` directory and load them using the :class:`~nlutestframework.implementations.simple_json_data_set.SimpleJSONDataSet` class.

**Step 3**: Prepare the NLU frameworks you want to benchmark. This framework has built-in support for a few NLU frameworks, so you might be able to skip this step. You can find the list of built-in frameworks :doc:`here <nlu_frameworks>`.

**Step 4**: Prepare a :ref:`configuration file <configuration-file>`. This repository comes with an example configuration, which benchmarks a few NLU frameworks that can run locally without any further configuration. You can find it in the `examples` directory. The example configuration also contains a few commented-out sections for NLU frameworks that run remotely and require additional configuration.

**Step 5**: Run the benchmark. The :doc:`installation <installation>`-step installed a script called ``nlutestframework``. Run ``nlutestframework`` in the ``examples`` directory to run the benchmark using the example configuration. ``nlutestframework --help`` gives you information about the arguments accepted by the script.

.. Note:: You can also start the benchmark programmatically by calling the respective methods of the :class:`~nlutestframework.nlu_benchmarker.NLUBenchmarker` class.
