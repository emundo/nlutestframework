import importlib
import os
import sys

# This setting prevents cut-off labels in plots created by matplotlib
from matplotlib import rcParams; rcParams.update({ "figure.autolayout": True }); del rcParams
import matplotlib.pyplot as plt

import yaml

from .global_config import GlobalConfig
from .has_logger import HasLogger
from .parallel_exception import run_in_parallel

# Other imports only for the type hints
from typing import Tuple, Dict, List, ClassVar, Optional, Any, NamedTuple
from .types import ConfusionMatrix, Intent, DataSetTitle, FrameworkTitle, JSONSerializable
from .nlu_data_set import NLUDataSet
from .nlu_framework import NLUFramework

class _Performance(NamedTuple):
    mean: float
    variance: float

class NLUBenchmarker(HasLogger):
    __instance: ClassVar["NLUBenchmarker"]

    @classmethod
    def getInstance(cls) -> "NLUBenchmarker":
        """
        Returns:
            The singleton instance of this class.
        """

        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = cls()
            cls.__cancel_flag = False
            return cls.__instance

    def cancel(self) -> None:
        """
        Abort the benchmark gracefully in the next situation possible.
        """

        self.__cancel_flag = True

    @staticmethod
    def __safeDivide(dividend: float, divisor: float) -> float:
        """
        Returns 0 for 0/0, the normal quotient otherwise.
        """

        if dividend == 0 and divisor == 0:
            return 0

        return dividend / divisor

    @staticmethod
    def __prepareF1Score(
        intent: Intent,
        confusion_matrix: ConfusionMatrix
    ) -> Tuple[int, int, int, int]:
        true_positives  = 0
        true_negatives  = 0
        false_positives = 0
        false_negatives = 0

        for expected, actual in confusion_matrix.items():
            for detected, amount in actual.items():
                if intent == expected:
                    if detected == intent:
                        true_positives += amount
                    else:
                        false_negatives += amount
                else:
                    if detected == intent:
                        false_positives += amount
                    else:
                        true_negatives += amount

        return (true_positives, true_negatives, false_positives, false_negatives)

    @classmethod
    def confusionMatrixToF1Scores(cls, confusion_matrix: ConfusionMatrix) -> Dict[Intent, float]:
        """
        Args:
            confusion_matrix: The confusion matrix to calculate F1 scores from.

        Returns:
            A mapping from intents to their respective F1 scores.
        """

        f1_scores = {}

        for intent in confusion_matrix.keys():
            true_positives, _, false_positives, false_negatives = cls.__prepareF1Score(
                intent,
                confusion_matrix
            )

            precision = cls.__safeDivide(true_positives, true_positives + false_positives)
            recall    = cls.__safeDivide(true_positives, true_positives + false_negatives)

            # Note: The value is multiplied times 100 here, which is not the standard for F1 scores.
            # This is to get a more intuitive score that is (roughly) between 0 and 100. It also
            # makes variances more graspable.
            f1_score = cls.__safeDivide(100 * 2 * precision * recall, precision + recall)

            f1_scores[intent] = f1_score

        return f1_scores

    @staticmethod
    def __plot(
        performances: Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, _Performance]]]
    ) -> None:
        # Create one chart per data set
        for framework_performances in performances.values():
            # Collect all intents that the frameworks were evaluated on
            intents: List[Intent] = []
            for intent_performances in framework_performances.values():
                intents.extend(intent_performances.keys())

            # Filter out the None-intent, because None can't be sorted together with strings
            intents = list(filter(lambda x: x is not None, intents))

            # Remove duplicates and sort alphabetically
            intents = sorted(set(intents))

            # Append the None-intent again, as the last entry
            intents.append(None)

            # Convert None to the string "None"
            intent_labels = list(map(str, intents))

            for framework_title, intent_performances in framework_performances.items():
                scores: List[Optional[float]] = []
                for intent in intents:
                    try:
                        scores.append(intent_performances[intent].mean)
                    except KeyError:
                        scores.append(None)

                plt.plot(intent_labels, scores, label="${}$".format(str(framework_title)))

            plt.legend(loc="upper left")
            plt.ylabel("F1 score * 100")
            plt.xlabel("Intents")
            plt.xticks(rotation=90)

            plt.show()

    async def __run(
        self,
        frameworks: List[NLUFramework],
        data_sets: List[NLUDataSet],
        num_iterations: int
    ) -> List[Dict[DataSetTitle, Dict[FrameworkTitle, ConfusionMatrix]]]:
        """
        Run n iterations of benchmarking for each framework on each data set. The benchmark results
        are returned "raw", that is one confusion matrix for each framework on each data set for
        each iteration.

        Args:
            frameworks: The frameworks to benchmark.
            data_sets: The data sets to benchmark on.
            num_iterations: The number of iterations to repeat the evaluation process.

        Returns:
            The performance of each framework on each data set over n iterations.
        """

        # Mypy requires the repetition of the return type
        iterations: List[Dict[DataSetTitle, Dict[FrameworkTitle, ConfusionMatrix]]] = []

        # Run the evaluation, one data set after another
        for data_set in data_sets:
            self._logger.info("Data set \"%s\"", data_set.title)

            await run_in_parallel(
                frameworks,
                # How could mypy possibly know the type of this lamda?
                lambda x, data_set=data_set: x.prepareDataSet(data_set), # type: ignore
                lambda x, _: x.unprepareDataSet(),
                "Error while preparing all frameworks for the {} data set.".format(data_set.title)
            )

            try:
                if self.__cancel_flag:
                    raise KeyboardInterrupt

                for i in range(num_iterations):
                    self._logger.info("\tIteration %d", i + 1)

                    performances = await run_in_parallel(
                        frameworks,
                        # Another one of these impossible-to-infer lambdas
                        lambda x, data_set=data_set: x.benchmark(data_set), # type: ignore
                        None,
                        "Error while benchmarking all frameworks."
                    )

                    data_set.reshuffle()

                    if len(iterations) == i:
                        iterations.append({})
                    iterations[i][data_set.title] = {
                        framework.title: performance
                        for framework, performance
                        in zip(frameworks, performances)
                    }

                    if self.__cancel_flag:
                        raise KeyboardInterrupt
            finally:
                # Make sure to always give the frameworks the chance to unprepare, even if something
                # went wrong.
                await run_in_parallel(
                    frameworks,
                    lambda x: x.unprepareDataSet(),
                    None,
                    "Error unpreparing all frameworks."
                )

        return iterations

    @classmethod
    def __confusionMatricesToF1Scores(
        cls,
        iterations: List[Dict[DataSetTitle, Dict[FrameworkTitle, ConfusionMatrix]]]
    ) -> List[Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, float]]]]:
        """
        Calculate F1 scores from confusion matrices.
        """

        iterations_result = []

        for iteration in iterations:
            iteration_result = {}

            for data_set_title, performances in iteration.items():
                performances_result = {}

                for framework_title, confusion_matrix in performances.items():
                    performances_result[framework_title] = cls.confusionMatrixToF1Scores(
                        confusion_matrix
                    )

                iteration_result[data_set_title] = performances_result

            iterations_result.append(iteration_result)

        return iterations_result

    @staticmethod
    def __f1ScoreMeansAndVariances(
        frameworks: List[NLUFramework],
        data_sets: List[NLUDataSet],
        iterations: List[Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, float]]]]
    ) -> Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, _Performance]]]:
        """
        Calculate F1 score means and variances for each intent over all iterations.
        """

        performances: Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, _Performance]]] = {}

        for data_set in data_sets:
            performances[data_set.title] = {}

            intents = { x.intent for x in data_set.validation_data }

            for framework in frameworks:
                performances[data_set.title][framework.title] = {}

                for intent in intents:
                    def score_getter(
                        x: Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, float]]],
                        data_set: NLUDataSet = data_set,
                        framework: NLUFramework = framework,
                        intent: Intent = intent
                    ) -> Optional[float]:
                        return x[data_set.title][framework.title].get(intent, None)

                    # mypy doesn't understand that you can call score_getter without passing all
                    # four arguments.
                    scores = list(map(
                        # Some intents may not have been part of the validation data in all
                        # runs. In these runs, the score is set to None and is excluded from the
                        # calculations later.
                        score_getter,
                        iterations
                    ))

                    # Remove None-scores, see comments above
                    filtered_scores: List[float] = list(filter(
                        lambda x: x is not None,
                        scores # type: ignore
                    ))

                    # If an intent was not included in any of the validation data, ignore the
                    # intent completely.
                    if len(filtered_scores) == 0:
                        continue

                    mean     = sum(filtered_scores) / len(filtered_scores)
                    variance = sum(map(
                        lambda x, mean=mean: (x - mean) ** 2, # type: ignore
                    filtered_scores)) / len(filtered_scores)

                    performances[data_set.title][framework.title][intent] = _Performance(
                        mean     = mean,
                        variance = variance
                    )

        return performances

    @staticmethod
    def __mergeIntentPerformances(
        performances: Dict[DataSetTitle, Dict[FrameworkTitle, Dict[Intent, _Performance]]]
    ) -> Dict[DataSetTitle, Dict[FrameworkTitle, _Performance]]:
        """
        Calculate the average performances across all intents for each data set and each framework.
        """

        # Mypy requires the repetition of the return type
        result: Dict[DataSetTitle, Dict[FrameworkTitle, _Performance]] = {}

        for data_set_title, framework_performances in performances.items():
            result[data_set_title] = {}

            for framework_title, intent_performances in framework_performances.items():
                means: List[float] = list(map(lambda x: x.mean, intent_performances.values()))
                variances: List[float] = list(map(
                    lambda x: x.variance,
                    intent_performances.values()
                ))

                result[data_set_title][framework_title] = _Performance(
                    mean     = sum(means)     / len(means),
                    variance = sum(variances) / len(variances)
                )

        return result

    @staticmethod
    def __sortPerformances(
        performances: Dict[DataSetTitle, Dict[FrameworkTitle, _Performance]]
    ) -> Dict[DataSetTitle, List[Tuple[FrameworkTitle, _Performance]]]:
        """
        Sort the frameworks by their performances, best to worst, looking only at the mean.
        """

        result: Dict[DataSetTitle, List[Tuple[FrameworkTitle, _Performance]]] = {}

        for data_set_title, framework_performances in performances.items():
            result[data_set_title] = sorted(
                framework_performances.items(),
                key=lambda x: x[1].mean,
                reverse=True
            )

        return result

    def __printWinner(
        self,
        performances: Dict[DataSetTitle, List[Tuple[FrameworkTitle, _Performance]]],
        data_sets: List[NLUDataSet]
    ) -> None:
        """
        Print the "winner" for each data set, together with the average mean and variance.
        """

        longest_data_set_title_length = max(map(lambda x: len(x.title), data_sets))
        self._logger.info("Best performing frameworks for each data set:")
        for data_set_title, framework_performances in performances.items():
            self._logger.info(
                "\t%s : %s; Average performance: %6.2f (variance: %6.2f)",
                data_set_title.ljust(longest_data_set_title_length),
                framework_performances[0][0],
                framework_performances[0][1].mean,
                framework_performances[0][1].variance
            )

    async def run(
        self,
        frameworks: List[NLUFramework],
        data_sets: List[NLUDataSet],
        num_iterations: int
    ) -> None:
        """
        Measure the performance of each framework on each data set. Outputs a summary about which
        framework performed best on each data set; also generates charts with more details about all
        performances.

        This method guarantees that all frameworks are destroyed before returning.

        Args:
            frameworks: The frameworks to benchmark.
            data_sets: The data sets to benchmark on.
            num_iterations: The number of iterations to repeat the evaluation process. The result is
                the average over all iterations.
        """

        try:
            iterations = await self.__run(frameworks, data_sets, num_iterations)
        finally:
            # Make sure that the frameworks are destructed even if something goes wrong during the
            # benchmarking.
            await run_in_parallel(
                frameworks,
                lambda x: x.destruct(),
                None,
                "Error deconstructing all frameworks."
            )

        iterations_  = self.__confusionMatricesToF1Scores(iterations)
        performances = self.__f1ScoreMeansAndVariances(frameworks, data_sets, iterations_)

        self.__plot(performances)

        performances_  = self.__mergeIntentPerformances(performances)
        performances__ = self.__sortPerformances(performances_)

        self.__printWinner(performances__, data_sets)

    async def createFrameworks(
        self,
        global_config: GlobalConfig,
        framework_configs: Dict[FrameworkTitle, JSONSerializable]
    ) -> List[NLUFramework]:
        """
        Create and prepare :class:`~nlutestframework.nlu_framework.NLUFramework` s based on
        configuration dictionaries.

        This method guarantees that either all frameworks are created or none of them. That means,
        if something goes wrong in the progress, frameworks that already were created are destroyed
        again.

        Args:
            global_config: Global configuration for the whole test framework.
            framework_configs: A dictionary containing the configuration of the frameworks to
                create. See :ref:`configuration-framework` for more information.

        Raises:
            :exc:`ParallelException`: if at least one framework creation failed.
        """

        async def create_framework(
            title: FrameworkTitle,
            framework_config: JSONSerializable
        ) -> NLUFramework:
            # Split at the last dot, which should result in the package name and the class name
            # mypy is technically right that the framework_config, which is a JSONSerializable, is
            # not guaranteed to be a Dict here.
            module, class_name = framework_config["class"].rsplit(".", 1) # type: ignore

            # Remove the "class" key from the framework config
            del framework_config["class"] # type: ignore

            # Dynamically import the class and get a reference
            cls = getattr(importlib.import_module(module), class_name)

            # Create the framework instance
            return await cls.create(global_config, framework_config, title) # type: ignore

        return await run_in_parallel( # type: ignore
            framework_configs.items(), # type: ignore
            lambda x: create_framework(*x),
            lambda _, x: x.destruct(),
            "Failed to create framework instances."
        )

    @staticmethod
    def loadDataSets(
        configs: Dict[DataSetTitle, JSONSerializable],
        global_ignore_cache: bool = False
    ) -> List[NLUDataSet]:
        """
        Load data sets based on the configuration dictionary.

        Args:
            configs: A dictionary containing the configuration of the data sets to load. See
                :ref:`configuration-data-set` for more information.
            global_ignore_cache: A boolean indicating whether caches should be ignored globally.
                Defaults to False.
        """

        instances = []
        for title, config in configs.items():
            # Split at the last dot, which should result in the package name and the class name
            module, class_name = config["class"].rsplit(".", 1) # type: ignore

            # Remove the "class" key from the config
            del config["class"] # type: ignore

            # Dynamically import the class and get a reference
            cls = getattr(importlib.import_module(module), class_name)

            # If the cache is disabled globally, override the setting in the config
            if global_ignore_cache:
                config["ignore_cache"] = True # type: ignore

            # Create the data set instance
            instances.append(cls(title, **config))

        return instances

    async def runFromConfig(
        self,
        config: Dict[str, Dict[str, JSONSerializable]],
        **global_config_override: Any
    ) -> None:
        """
        Load and run a full benchmark from a single configuration dictionary.

        Args:
            config: A dictionary containing the full configuration required to load and run the
                benchmark. See :ref:`configuration-full` for more information.
            **global_config_override: Options to override in the global configuration.
        """

        global_config = {
            "python"       : config["global"].get("python", sys.executable),
            "iterations"   : config["global"]["iterations"],
            "ignore_cache" : config["global"].get("ignore_cache", False)
        }
        global_config.update(global_config_override)
        global_config_ = GlobalConfig(**global_config) # type: ignore

        # Load the data sets first, so that the frameworks don't have to be destroyed if loading the
        # data sets fails.
        data_sets  = self.loadDataSets(config["data_sets"], global_config_.ignore_cache)
        frameworks = await self.createFrameworks(global_config_, config["frameworks"])

        await self.run(frameworks, data_sets, global_config_.iterations)

    async def runFromConfigFile(self, path: str, **global_config_override: Any) -> None:
        """
        Load and run a full benchmark from a configuration file.

        Args:
            path: The path to the configuration file. See :ref:`configuration-file` for more
                information.
            **global_config_override: Options to override in the global configuration.

        Raises:
            :exc:`OSError`: in case the config file could not be read due to I/O or other OS-related
                issues.
            :exc:`yaml.YAMLError`: in case the config file contains invalid YAML.
        """

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Data set configuration file convenience: The data paths may be given relative to the
        # configuration file location.
        for data_set_config in config["data_sets"].values():
            # Make the data paths absolute, relative to the location of the configuration file.
            if not os.path.isabs(data_set_config["data_path"]):
                data_set_config["data_path"] = os.path.abspath(os.path.join(
                    os.path.dirname(path),
                    data_set_config["data_path"]
                ))

        # Data set convenience: Implementations included in this library don't have to be specified
        # using the whole package name.
        for data_set_config in config["data_sets"].values():
            if "." not in data_set_config["class"]:
                data_set_config["class"] = "nlutestframework.implementations.{}DataSet".format(
                    data_set_config["class"]
                )

        # Framework convenience: Implementations included in this library don't have to be specified
        # using the whole package name.
        for framework_config in config["frameworks"].values():
            if "." not in framework_config["class"]:
                framework_config["class"] = (
                    "nlutestframework.implementations.{}NLUFramework".format(
                        framework_config["class"]
                    )
                )

        await self.runFromConfig(config, **global_config_override)
