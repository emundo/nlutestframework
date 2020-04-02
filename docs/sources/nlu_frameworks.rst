List of Built-In NLU Frameworks
===============================

+-----------------------------------------------------+--------------------------------------------------------------------------------------------+------------------------+
| NLU Framework                                       | Class                                                                                      | Threshold Optimizable? |
+=====================================================+============================================================================================+========================+
| `Google Dialogflow <https://dialogflow.com/>`_      | :class:`~nlutestframework.implementations.dialogflow_nlu_framework.DialogflowNLUFramework` | No                     |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+------------------------+
| `Microsoft LUIS <https://www.luis.ai/home>`_        | :class:`~nlutestframework.implementations.luis_nlu_framework.LUISNLUFramework`             | Yes                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+------------------------+
| `Rasa NLU <https://rasa.com/docs/rasa/nlu/about/>`_ | :class:`~nlutestframework.implementations.rasa_nlu_framework.RasaNLUFramework`             | Yes                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+------------------------+
| `Snips NLU <https://snips-nlu.readthedocs.io/>`_    | :class:`~nlutestframework.implementations.snips_nlu_framework.SnipsNLUFramework`           | No                     |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+------------------------+

Configuration
-------------

Each NLU framework implementation provides a different set of configuration options. Refer to the :meth:`~nlutestframework.nlu_framework.NLUFramework.construct` method of the respective implementation for the available options.

Some of the frameworks don't support the concept of a None-intent or implement it in a way that is not compatible with how this framework handles it. These frameworks subclass the :class:`~nlutestframework.optimizable_nlu_framework.OptimizableNLUFramework` class and provide additional configuration options regarding the optional optimization for a threshold-based implementation of the None-intent. Refer to the API documentation of the :class:`~nlutestframework.optimizable_nlu_framework.OptimizableNLUFramework` class for details on the optimization process and the available options.

Additional Steps
----------------

Some frameworks require additional steps to get them up and running.

Google Dialogflow
^^^^^^^^^^^^^^^^^

To benchmark Google Dialogflow, a Dialogflow/Google project has to be provided to the test framework. The credentials have to be provided by pointing the environment variable ``GOOGLE_APPLICATION_CREDENTIALS`` to the json file containing the credentials. Additional options required by the :class:`~nlutestframework.implementations.dialogflow_nlu_framework.DialogflowNLUFramework` class have to be provided in :doc:`the configuration <configuration>`.

.. Warning:: The agent is deleted/re-created during the benchmarking progress, all data present before the benchmark is lost. If the agent contains important data, make sure to `export the agent <https://cloud.google.com/dialogflow/docs/agents-settings#export>`_ beforehand.

Microsoft LUIS
^^^^^^^^^^^^^^

To benchmark Microsoft LUIS, a LUIS project has to be provided to the test framework. To do so, the options required by the :class:`~nlutestframework.implementations.luis_nlu_framework.LUISNLUFramework` class have to be provided in :doc:`the configuration <configuration>`. Make sure that the application name ``NLUTestFramework`` is unused in your LUIS instance.

Rasa NLU
^^^^^^^^

To benchmark Rasa NLU, the benchmarking host and the executing user need access to a Docker daemon. Other than that no setup is required, all interactions with Docker are fully automated.

Snips NLU
^^^^^^^^^

Snips NLU doesn't require any setup.
