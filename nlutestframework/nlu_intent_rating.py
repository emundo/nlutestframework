# Other imports only for the type hints
from typing import List, Tuple
from .types import Intent

class NLUIntentRating:
    def __init__(self, sentence: str, rated_intents: List[Tuple[Intent, float]]):
        """
        Args:
            sentence: The sentence rated by the NLU framework.
            rated_intents: A mapping from intents to floats encoding the confidence of the NLU
                framework, that the rated sentence belongs to that intent. The confidence ranges
                from 0 to 1.
        """

        self.__sentence = sentence
        self.__sorted_intents = sorted(rated_intents, key=lambda x: x[1], reverse=True)

    @property
    def sentence(self) -> str:
        return self.__sentence

    @property
    def sorted_intents(self) -> List[Tuple[Intent, float]]:
        """
        Returns:
            A copy of the mapping passed to the constructor, but sorted from highest confidence to
            lowest.
        """

        return list(self.__sorted_intents)

    @property
    def detected_intent(self) -> Intent:
        """
        Returns:
            The intent that was rated by the NLU framework with highest confidence.
        """

        return self.__sorted_intents[0][0]

    def noneIfBelow(self, threshold: float) -> None:
        """
        Replace the best-rated intent with the None-intent, if the confidence of the best-rated
        intent is below (or equal to) a certain threshold.

        Args:
            threshold: The threshold.
        """

        # If the highest confidence is below (or equal to) the threshold...
        if self.__sorted_intents[0][1] <= threshold:
            # ...prepend the None-intent and full confidence to the rating list.
            self.__sorted_intents.insert(0, (None, 1.0))

    def __str__(self) -> str:
        intents = ""

        longest_intent_length = max(map(lambda x: len(str(x[0])), self.__sorted_intents))

        for intent, rating in self.__sorted_intents:
            intents += "\t\t{} : {}\n".format(str(intent).ljust(longest_intent_length), rating)

        return (
              "Intent rating for sentence \"{}\":\n".format(self.__sentence)
            + "\tIntents (sorted in descending order by rating):\n"
            + intents
        )
