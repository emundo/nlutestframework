import json
import os
import random

from langcodes import Language

from .has_logger import HasLogger
from .nlu_data_entry import NLUDataEntry

# Other imports only for the type hints
from typing import Optional, List
from .types import DataSetTitle

class NLUDataSet(HasLogger):
    def __init__(
        self,
        title: DataSetTitle,
        data_path: str,
        validation_percentage: int,
        language: Optional[str] = None,
        ignore_cache: bool = False
    ):
        """
        Args:
            title: The title of this data set.
            data_path: The path to the file or directory containing the data in an
                implementation-specific format. User directory references and environment variables
                are expanded.
            validation_percentage: The percentage of the data to be used for performance validation,
                the remaining data is used for training. Expects a positive whole number between 0
                and 100.
            language: The language tag of this data set, e.g. "en" or "en-us". If this parameter is
                set to None or omitted, the implementation is assumed to get that information from
                somewhere else.
            ignore_cache: A boolean flag indicating whether the data cache should be ignored.

        Raises:
            :exc:`OSError`: in case the data could not be loaded or cached due to I/O or other
                OS-related issues.
            :exc:`ValueError`: if the validation data set or the training data set are empty after
                splitting the data based on validation_percentage.
            :exc:`ValueError`: if the data path does not point to an existing file or directory.

        Sentences assigned to the None-intent are treated differently. These sentences are first
        removed from the data set, the remaining data is then shuffled and split and the None-data
        is added to the validation data in the final step.
        """

        super().__init__()

        self.__title    = title
        self.__language = None

        if language is not None:
            self._setLanguage(language)

        # Expand user directory references, environment variables and make the path absolute.
        data_path = os.path.abspath(os.path.expandvars(os.path.expanduser(data_path)))

        if not os.path.exists(data_path):
            raise ValueError("The data path does not point to an existing file or directory.")

        # Try to load the cached data
        if not (ignore_cache or self.__loadCachedData(data_path)):
            ignore_cache = True

        # If the cache was ignored (also set if loading the cache failed), load the data "by hand"
        if ignore_cache:
            data = self._loadData(data_path)

            # Only the data without None-intent
            self.__data = list(filter(lambda x: x.intent is not None, data))

            # Only the data with None-intent
            self.__none_data = list(filter(lambda x: x.intent is None, data))

            self.__cacheData(data_path)

        # Apply the split percentage only to the data without None-intent
        self.__validation_size = (validation_percentage * len(self.__data)) // 100

        # Perform an initial shuffle-and-split
        self.reshuffle()

    @property
    def title(self) -> str:
        return self.__title

    @property
    def language(self) -> str:
        return self.__language # type: ignore

    def _setLanguage(self, language: str) -> None:
        """
        Args:
            language: The language of this data set. Use this method to set the language, if the
                language was dynamically loaded from the data set itself. The language is
                represented by its ISO 639-1 code (e.g. "en").

        Raises:
            :exc:`ValueError`: if the language was already set.
        """

        if self.__language is not None:
            raise ValueError("The language for this data set was already set.")

        self.__language = Language.get(language).maximize().to_tag()

    def __loadCachedData(self, data_path: str) -> bool:
        """
        Args:
            data_path: The absolute path to the file or directory containing the original data.

        Returns:
            A boolean indicating whether loading the cache was successful.
        """

        if os.path.isfile(data_path):
            cache_file = data_path + ".nlucache"
        else:
            cache_file = os.path.join(data_path, "nlu.cache")

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Load the language and make sure it is consistent with the expected language
            language = data["language"]
            if self.__language is None or self.__language == language:
                self.__language = language
            else:
                raise ValueError("Language clash between cached data and expected language.")

            # Load the entries
            entries = [ NLUDataEntry.fromSerialized(entry) for entry in data["entries"] ]

            # Only the data without None-intent
            self.__data = list(filter(lambda x: x.intent is not None, entries))

            # Only the data with None-intent
            self.__none_data = list(filter(lambda x: x.intent is None, entries))

            return True
        except BaseException as e: # pylint: disable=broad-except
            self._logger.warning("Error loading cached data", exc_info=e)
            return False

    def __cacheData(self, data_path: str) -> None:
        """
        Args:
            data_path: The absolute path to the file or directory containing the original data.

        Raises:
            :exc:`OSError`: if storing the cache fails.
        """

        if os.path.isfile(data_path):
            cache_file = data_path + ".nlucache"
        else:
            cache_file = os.path.join(data_path, "nlu.cache")

        entries = self.__data + self.__none_data

        data = {
            "language" : self.__language,
            "entries"  : [ entry.serialize() for entry in entries ]
        }

        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _loadData(self, data_path: str) -> List[NLUDataEntry]:
        """
        Args:
            data_path: The absolute path to the file or directory containing the original data in an
                implementation-specific format.

        Returns:
            A list of NLUDataEntry instances.

        Raises:
            :exc:`OSError`: in case the data could not be loaded due to I/O or other OS-related
                issues.
        """

        raise NotImplementedError("To be implemented by subclasses.")

    def reshuffle(self) -> None:
        """
        Shuffle the data and split it into training and validation data. This method does not make
        sure that the new splitting is different from previous splittings, but given a decent amount
        of data the chance for that should be low enough.

        Raises:
            :exc:`ValueError`: if the validation data set or the training data set are empty after
                splitting the data.
        """

        # Shuffle the data without None-intent
        random.shuffle(self.__data)

        # Split the data without None-intent into training and validation data
        self.__training   = self.__data[self.__validation_size:]
        self.__validation = self.__data[:self.__validation_size]

        # Make sure that both sets are non-empty
        if len(self.__validation) == 0 or len(self.__training) == 0:
            raise ValueError("Validation or training data is empty.")

        # Extend the validation data by the None-intent data
        self.__validation.extend(self.__none_data)

    @property
    def training_data(self) -> List[NLUDataEntry]:
        """
        Returns:
            The data to train on.
        """

        return list(self.__training)

    @property
    def validation_data(self) -> List[NLUDataEntry]:
        """
        Returns:
            The data to validate with.
        """

        return list(self.__validation)

    def __str__(self) -> str:
        return "NLU data set \"{}\" with {} entries.".format(self.title, len(self.__data))
