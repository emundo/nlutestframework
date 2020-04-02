from .has_logger import HasLogger

# Other imports only for the type hints
from typing import List, Dict, Any
from .types import FrameworkTitle, JSONSerializable, ConfusionMatrix
from .global_config import GlobalConfig
from .nlu_data_entry import NLUDataEntry
from .nlu_data_set import NLUDataSet
from .nlu_intent_rating import NLUIntentRating

class NLUFramework(HasLogger):
    """
    Frameworks follow a certain lifecycle while being benchmarked. Implementations can react to new
    stages of their lifecycle by overriding methods provided by this base class.

    The lifecycle and the corresponding event methods look like this::

        .construct(...)
            for each data set:
                .prepareDataSet(...)
                    repeat n times:
                        .train(...)
                        # Validation
                        .cleanupTraining(...)
                .unprepareDataSet(...)
        .deconstruct(...)

    Frameworks are not forced to react to all lifecycle events.

    See :doc:`../nlu_frameworks`.
    """

    # This is just to satisfy mypy. Please don't call it directly!
    def __init__(self, *args: Any, **kwargs: Any):
        self.__title: str

        super().__init__(*args, **kwargs)

    @property
    def title(self) -> FrameworkTitle:
        return self.__title

    # Asynchronous replacement for the usual __init__ constructor
    @classmethod
    async def create(
        cls,
        global_config: GlobalConfig,
        framework_config: Dict[str, JSONSerializable],
        title: FrameworkTitle
    ) -> "NLUFramework":
        """
        Construct an instance of this NLUFramework implementation.

        Args:
            global_config: Global configuration for the whole test framework.
            framework_config: A dictionary containing configuration options specific to this
                :class:`NLUFramework` implementation. See the specific implementation of
                :meth:`construct` for more details.
            title: The title of this framework.

        Returns:
            A new and prepared instance of this class.

        Raises:
            :exc:`TypeError`: if the framework configuration is incomplete or malformed.
        """

        instance = cls()
        instance.__title = title # pylint: disable=protected-access
        await instance.construct(global_config, **framework_config)
        return instance

    async def construct(
        self,
        global_config: GlobalConfig,
        **framework_config: JSONSerializable
    ) -> None:
        """
        Args:
            global_config: Global configuration for the whole test framework.
            **framework_config: Keyword arguments containing configuration options specific to this
                :class:`NLUFramework` implementation. See the specific implementation for more
                details.
        """

        pass

    async def destruct(self) -> None:
        """
        This method gives framework implementations the opportunity to execute final cleanup steps
        before the framework exits.
        """

        pass

    # pylint: disable=attribute-defined-outside-init
    async def prepareDataSet(self, data_set: NLUDataSet) -> None:
        """
        Args:
            data_set: The data set that will be benchmarked next.
        """

        pass

    async def unprepareDataSet(self) -> None:
        """
        This method gives framework implementations the opportunity to execute cleanup steps before
        preparing for a new data set (or exiting).
        """

        pass

    # pylint: disable=attribute-defined-outside-init
    async def train(self, training_data: List[NLUDataEntry]) -> None:
        """
        Args:
            training_data: The data to train on. Must not be empty.
        """

        raise NotImplementedError("To be implemented by subclasses.")

    async def rateIntents(self, sentence: str) -> NLUIntentRating:
        """
        Args:
            sentence: A sentence to find intents and entities for.

        Returns:
            The intent rating information as returned by the framework.
        """

        raise NotImplementedError("To be implemented by subclasses.")

    async def cleanupTraining(self) -> None:
        """
        Perform cleanup on the NLU framework. For example, this can include resetting the framework
        to a pre-training state.
        """

        raise NotImplementedError("To be implemented by subclasses.")

    async def __validate(self, validation_data: List[NLUDataEntry]) -> ConfusionMatrix:
        """
        Args:
            validation_data: The data to validate the NLU framework against. Must not be empty.

        Returns:
            The validation results encoded in a confusion matrix.
        """

        confusion_matrix: ConfusionMatrix = {}

        for datum in validation_data:
            confusion_matrix[datum.intent] = confusion_matrix.get(datum.intent, {})

            rating = await self.rateIntents(datum.sentence)

            confusion_matrix[datum.intent][rating.detected_intent] = (
                confusion_matrix[datum.intent].get(rating.detected_intent, 0) + 1
            )

        return confusion_matrix

    async def benchmark(self, data_set: NLUDataSet) -> ConfusionMatrix:
        """
        Benchmark this NLU framework on the given data. This method starts by training the
        framework, followed by measuring the performance of the framework and finished by cleaning
        up whatever needs to be cleaned.

        Args:
            data_set: The data set to benchmark on.

        Returns:
            The validation results encoded in a confusion matrix.
        """

        self._logger.info(
            "Benchmarking framework \"%s\" on data set \"%s\"",
            self.__title,
            data_set.title
        )

        try:
            await self.train(data_set.training_data)
            performance = await self.__validate(data_set.validation_data)
        finally:
            # Guarantee the cleanup
            await self.cleanupTraining()

        return performance
