Configuration
=============

The framework requires three types of configuration:

- Global configuration, as explained :ref:`here <configuration-global>`
- Data set configuration, as explained :ref:`here <configuration-data-set>`
- Framework configuration, as explained :ref:`here <configuration-framework>`

The configuration can either be passed to the :class:`~nlutestframework.nlu_benchmarker.NLUBenchmarker` directly or loaded from a :ref:`configuration file <configuration-file>`.

.. _configuration-global:

Global Configuration
--------------------

Configuration which is global to the test framework. Refer to the :meth:`GlobalConfig <nlutestframework.global_config.GlobalConfig.__init__>` class for supported options.

.. _configuration-data-set:

Data Set Configuration
----------------------

Configuration for a single data set. The :ref:`full configuration <configuration-full>` may contain multiple of these. Refer to the :meth:`NLUDataSet <nlutestframework.nlu_data_set.NLUDataSet.__init__>` class for supported options.

.. _configuration-framework:

Framework Configuration
-----------------------

Configuration for a single NLU framework. The :ref:`full configuration <configuration-full>` may contain multiple of these. In contrast to the :ref:`global configuration <configuration-global>` and the :ref:`data set configuration <configuration-data-set>`, the available configuration options vary between implementations. Refer to the :doc:`list of built-in implementations <nlu_frameworks>` for details on the configuration and setup required for each framework.

.. _configuration-full:

Full Configuration
------------------

The following example shows the structure of a full configuration:

.. code-block:: JSON

    {
        "global": {
            "iterations": 10
        },
        "data_sets": {
            "AskUbuntuCorpus": {
                "class": "nlutestframework.implementations.SimpleJSONDataSet",
                "data_path": "/path/to/nlutestframework/data/corpora/AskUbuntuCorpus.json",
                "validation_percentage": 50
            },
            "ChatbotCorpus": {
                "class": "nlutestframework.implementations.SimpleJSONDataSet",
                "data_path": "/path/to/nlutestframework/data/corpora/ChatbotCorpus.json",
                "validation_percentage": 50
            },
            "WebApplicationsCorpus": {
                "class": "nlutestframework.implementations.SimpleJSONDataSet",
                "data_path": "/path/to/nlutestframework/data/corpora/WebApplicationsCorpus.json",
                "validation_percentage": 50
            }
        },
        "frameworks": {
            "Rasa NLU": {
                "class": "nlutestframework.implementations.RasaNLUFramework",
                "pipeline": "pretrained"
            },
            "Snips NLU": {
                "class": "nlutestframework.implementations.SnipsNLUFramework"
            }
        }
    }

The configuration is a dictionary which consists of the three sections ``global``, ``data_sets`` and ``frameworks``.

The ``global`` section simply contains the :ref:`global configuration <configuration-global>`.

The ``data_sets`` and ``frameworks`` sections each contain multiple configurations. The keys denote the name of the respective data set or framework, while the values contain the actual configuration. There is one special configuration option, which is common to all data sets and frameworks: the Python-``class`` to use for that specific data set or framework.

.. _configuration-file:

Configuration Files
-------------------

Configuration files contain the same configuration as introduced :ref:`above <configuration-full>`, but formatted using YAML.

Relative paths are converted to absolute paths based on the location of the configuration file.

For convenience, when refering to classes in the :mod:`nlutestframework.implementations` package, the full package name is not required. For example passing ``Snips`` as the class for a framework is equivalent to passing ``nlutestframework.implementations.SnipsNLUFramework``.

A full example for a file ``config.yml``:

.. literalinclude:: ../../examples/config.yml
    :language: YAML
