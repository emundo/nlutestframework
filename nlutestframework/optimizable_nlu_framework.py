from .intent_threshold_optimizer import IntentThresholdOptimizer
from .nlu_framework import NLUFramework

# Other imports only for the type hints
from .nlu_data_set import NLUDataSet
from .nlu_intent_rating import NLUIntentRating

class OptimizableNLUFramework(NLUFramework):
    """
    Base class for NLU frameworks that need a threshold-based implementation of the None-intent.
    Beware that some of the method names you have to implement differ from
    :class:`~nlutestframework.nlu_framework.NLUFramework`, namely :meth:`_prepareDataSet`
    (was: :meth:`~nlutestframework.nlu_framework.NLUFramework.prepareDataSet`) and
    :meth:`_rateIntents` (was: :meth:`~nlutestframework.nlu_framework.NLUFramework.rateIntents`).
    """

    # pylint: disable=arguments-differ
    async def construct( # type: ignore
        self,
        intent_threshold: float = 0.3,
        optimize_intent_threshold: bool = False,
        optimizer_iterations: int = 5,
        optimizer_grid_search_step_size: float = 0.01
    ) -> None:
        """
        Args:
            intent_threshold: Some frameworks don't have the None-intent or a comparable, compatible
                concept built-in. For these frameworks, this implementation selects the None-intent
                based on the highest confidence of all intents. If the highest confidence is below
                intent_threshold, the rated intent is interpreted as the None-intent instead.
                Confidence values range from 0 to 1, the default value of intent_threshold is 0.3.
            optimize_intent_threshold: When this flag is set, the value of :obj:`intent_threshold`
                is ignored. Instead, the threshold is automatically optimized for each data set
                during a few additional training iterations.
            optimizer_iterations: The number of iterations to repeat and average the threshold
                optimization. Defaults to 5.
            optimizer_grid_search_step_size: The step size for the grid search over the threshold.
                The optimal threshold is searched for in a window from 0 to 1, so a step size of
                e.g. 0.01 means that 100 different values are tested.
        """

        self.__intent_threshold = intent_threshold
        self.__optimize_intent_threshold = optimize_intent_threshold
        self.__optimizer_iterations = optimizer_iterations
        self.__optimizer_grid_search_step_size = optimizer_grid_search_step_size

    async def prepareDataSet(self, data_set: NLUDataSet) -> None:
        await self._prepareDataSet(data_set)

        if self.__optimize_intent_threshold:
            self.__intent_threshold = 0 # Set the threshold to 0 during the optimization
            self.__intent_threshold = await IntentThresholdOptimizer.optimize(
                self,
                data_set,
                self.__optimizer_iterations,
                self.__optimizer_grid_search_step_size
            )

    # pylint: disable=attribute-defined-outside-init
    async def _prepareDataSet(self, data_set: NLUDataSet) -> None:
        """
        Args:
            data_set: The data set that will be benchmarked next.
        """

        pass

    async def rateIntents(self, sentence: str) -> NLUIntentRating:
        rating = await self._rateIntents(sentence)

        rating.noneIfBelow(self.__intent_threshold)

        return rating

    async def _rateIntents(self, sentence: str) -> NLUIntentRating:
        """
        Args:
            sentence: A sentence to find intents and entities for.

        Returns:
            The intent rating information as returned by the framework.
        """

        raise NotImplementedError("To be implemented by subclasses.")
