import copy
import logging
import threading

from .nlu_benchmarker import NLUBenchmarker

# Other imports only for the type hints
from typing import Tuple, List, Dict
from .types import ConfusionMatrix, Intent
from .nlu_data_set import NLUDataSet
from .nlu_framework import NLUFramework
from .nlu_intent_rating import NLUIntentRating

# This setting prevents cut-off labels in plots created by matplotlib
from matplotlib import rcParams; rcParams.update({ "figure.autolayout": True }); del rcParams
import matplotlib.pyplot as plt

class IntentThresholdOptimizer:
    @classmethod
    async def optimize(
        cls,
        framework: NLUFramework,
        data_set: NLUDataSet,
        iterations: int,
        grid_step_size: float
    ) -> float:
        """
        Find the optimal confidence threshold for interpreting an intent classification result as
        the None-intent.

        The optimal threshold is found by running multiple iterations of the benchmark and searching
        for the threshold that maximizes the F1 score for each iteration. The optimal threshold is
        found by a grid search. The grids of each iteration are averaged and the threshold with the
        maximum F1 score found that way is returned.

        Args:
            framework: An NLU framework which is already prepared for the data set.
            data_set: The data set to optimize the threshold for.
            iterations: The number of iterations to repeat and average the threshold optimization.

        Returns:
            The optimized threshold.
        """

        grids: Dict[float, List[float]] = {}

        for i in range(iterations):
            logging.getLogger(cls.__name__).info(
                "Optimizing %s, iteration %s out of %s.",
                framework.title,
                i + 1,
                iterations
            )

            # Collect all grids
            grid = await cls.__iteration(framework, data_set, grid_step_size)
            for thresh, score in grid.items():
                grids[thresh] = grids.get(thresh, [])
                grids[thresh].append(score)

        grids_avg: Dict[float, Dict[str, float]] = {}

        # Calculate the means and variances of all entries in the grids
        for thresh, scores in grids.items():
            mean = sum(scores) / len(scores)
            var = sum(map(
                lambda x, mean=mean: (x - mean) ** 2, # type: ignore
            scores)) / len(scores)

            grids_avg[thresh] = { "mean": mean, "var": var }

        # Find the threshold with the highest F1 score
        t_max = { "score": 0., "thresh": 0. }
        for thresh, score_avg in grids_avg.items():
            if score_avg["mean"] > t_max["score"]:
                t_max = { "score": score_avg["mean"], "thresh": thresh }

        print("Threshold: F1 score mean (variance)")
        for thresh, score_avg in grids_avg.items():
            print("{:.2f}: {:.2f} ({:.2f})".format(thresh, score_avg["mean"], score_avg["var"]))

        print("Best: {:.2f}".format(t_max["thresh"]))

        cls.__plot(grids_avg, "Threshold Optimization of {} on {}".format(
            framework.title,
            data_set.title
        ))

        return t_max["thresh"]

    @classmethod
    async def __iteration(
        cls,
        framework: NLUFramework,
        data_set: NLUDataSet,
        grid_step_size: float
    ) -> Dict[float, float]:
        try:
            # Train
            await framework.train(data_set.training_data)

            # Classify (without applying any threshold)
            ratings = [
                (datum.intent, await framework.rateIntents(datum.sentence))
                for datum
                in data_set.validation_data
            ]
        finally:
            # Guarantee the cleanup
            await framework.cleanupTraining()

        # Re-shuffle the data set
        data_set.reshuffle()

        def calc_f1_for_threshold(threshold: float) -> float:
            thresholded_ratings = copy.deepcopy(ratings)
            for (_, rating) in thresholded_ratings:
                rating.noneIfBelow(threshold)

            confusion_matrix = cls.__toConfusionMatrix(thresholded_ratings)

            f1_scores = list(NLUBenchmarker.confusionMatrixToF1Scores(confusion_matrix).values())

            return sum(f1_scores) / len(f1_scores)

        # Run a grid search over the threshold
        grid: Dict[float, float] = {}
        thresh = 0.
        while thresh < 1.:
            grid[thresh] = calc_f1_for_threshold(thresh)

            thresh += grid_step_size

        return grid

    @staticmethod
    def __toConfusionMatrix(ratings: List[Tuple[Intent, NLUIntentRating]]) -> ConfusionMatrix:
        confusion_matrix: ConfusionMatrix = {}

        for (intent, rating) in ratings:
            confusion_matrix[intent] = confusion_matrix.get(intent, {})
            confusion_matrix[intent][rating.detected_intent] = (
                confusion_matrix[intent].get(rating.detected_intent, 0) + 1
            )

        return confusion_matrix

    @staticmethod
    def __plot(grids: Dict[float, Dict[str, float]], title: str) -> None:
        sorted_coord_pairs = sorted(grids.items(), key=lambda x: x[0])

        def _plot() -> None:
            plt.errorbar(
                [ x[0] for x in sorted_coord_pairs ],
                [ x[1]["mean"] for x in sorted_coord_pairs ],
                yerr=[ x[1]["var"] for x in sorted_coord_pairs ],
                capsize=5,
                capthick=2,
                errorevery=len(sorted_coord_pairs) // 20
            )

            plt.ylim(bottom=0)
            plt.ylabel("F1 score * 100 (mean over all iterations)")
            plt.xlabel("threshold")
            plt.title(title)
            plt.show()

        threading.Thread(target=_plot).start()
