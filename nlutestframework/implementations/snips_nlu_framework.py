import subprocess

from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import DEFAULT_CONFIGS

from ..nlu_framework import NLUFramework
from ..nlu_intent_rating import NLUIntentRating

from langcodes import Language

# Other imports only for the type hints
from typing import List
from ..global_config import GlobalConfig
from ..nlu_data_entry import NLUDataEntry
from ..nlu_data_set import NLUDataSet

class SnipsNLUFramework(NLUFramework):
    # pylint: disable=arguments-differ
    async def construct( # type: ignore
        self,
        global_config: GlobalConfig,
        skip_language_installations: bool = False
    ) -> None:
        """
        Args:
            global_config: Global configuration for the whole test framework.
            skip_language_installations: A boolean indicating whether to skip the installation of
                required language resources. Defaults to False.
        """

        self.__python = global_config.python
        self.__skip_language_installations = skip_language_installations

    # pylint: disable=attribute-defined-outside-init
    async def prepareDataSet(self, data_set: NLUDataSet) -> None:
        last_exception = None

        # Try all language tag derivations, from specific to broad
        for language in Language.get(data_set.language).simplify_script().broaden():
            language = language.to_tag()
            try:
                if not self.__skip_language_installations:
                    self._logger.info("Installing language resources for \"%s\"...", language)

                    subprocess.run(
                        [ self.__python, "-m", "snips_nlu", "download", language ],
                        check=True
                    )

                self.__language = language

                last_exception = None
                break
            except BaseException as e: # pylint: disable=broad-except
                last_exception = e

        if last_exception is not None:
            raise last_exception

    async def unprepareDataSet(self) -> None:
        del self.__language

    # pylint: disable=attribute-defined-outside-init
    async def train(self, training_data: List[NLUDataEntry]) -> None:
        self.__engine = SnipsNLUEngine(DEFAULT_CONFIGS[self.__language])

        intents = {}

        for intent in { x.intent for x in training_data }:
            utterances = []

            # Lambda not correctly supported in mypy
            for entry in filter(lambda x, i=intent: x.intent == i, training_data): # type: ignore
                utterances.append({ "data": [ { "text": entry.sentence } ] })

            intents[intent] = { "utterances": utterances }

        self.__engine.fit({
            "language" : self.__language,
            "intents"  : intents,
            "entities" : {}
        })

    async def rateIntents(self, sentence: str) -> NLUIntentRating:
        intents = self.__engine.get_intents(sentence)

        return NLUIntentRating(
            sentence,
            [ (x["intentName"], x["probability"]) for x in intents ]
        )

    async def cleanupTraining(self) -> None:
        del self.__engine
