import asyncio
import random
import string

import dialogflow_v2
from google.api_core.exceptions import FailedPrecondition, ResourceExhausted
from langcodes import Language

from ..nlu_framework import NLUFramework
from ..nlu_intent_rating import NLUIntentRating

# Other imports only for the type hints
from typing import List, Dict, Any
from ..global_config import GlobalConfig
from ..nlu_data_entry import NLUDataEntry
from ..nlu_data_set import NLUDataSet

class DialogflowNLUFramework(NLUFramework):
    # pylint: disable=arguments-differ
    async def construct( # type: ignore
        self,
        _global_config: GlobalConfig,
        time_zone: str,
        project: str = "nlutestframework",
        agent: str = "NLUTestFramework"
    ) -> None:
        """
        Args:
            _global_config: Global configuration for the whole test framework.
            time_zone: The time zone, e.g. Europe/Berlin. See https://www.iana.org/time-zones for
                the list of possible values.
            project: The name of the Dialogflow project. Defaults to "nlutestframework".
            agent: The name of the Dialogflow agent. Defaults to "NLUTestFramework".
        """

        self.__time_zone = time_zone
        self.__project   = project
        self.__agent     = agent

        # Create the various clients to interact with the Dialogflow API
        clients_config: Dict[str, Any] = {}

        self.__agents_client   = dialogflow_v2.AgentsClient(**clients_config)
        self.__intents_client  = dialogflow_v2.IntentsClient(**clients_config)
        self.__sessions_client = dialogflow_v2.SessionsClient(**clients_config)

        await self.__removeIntents()

    # pylint: disable=attribute-defined-outside-init
    async def prepareDataSet(self, data_set: NLUDataSet) -> None:
        self.__language = Language.get(data_set.language).simplify_script().to_tag()

        agent_parent_path = self.__agents_client.project_path(self.__project)

        # The default language code doesn't really matter as this code always explicitly passes the
        # exact language on each step. Still, the default language code HAS to be set and it MUST
        # be set to the code that already is the default.
        # The following code attempts to retrieve the current agent and to extract the current
        # default language code from it.
        try:
            default_language_code = self.__agents_client.get_agent(
                agent_parent_path
            ).default_language_code
        except: # pylint: disable=bare-except
            # TODO: Unable to figure out which exact error is raised in case the agent doesn't
            # exist, which is why this code catches any exception that might be raised by the call
            # to get_agent.
            default_language_code = "en"

        self.__agents_client.set_agent(dialogflow_v2.types.Agent(
            parent       = agent_parent_path,
            display_name = self.__agent,
            time_zone    = self.__time_zone,
            default_language_code    = default_language_code,
            supported_language_codes = [ self.__language ]
        ))

    async def unprepareDataSet(self) -> None:
        del self.__language

    # pylint: disable=attribute-defined-outside-init
    async def train(self, training_data: List[NLUDataEntry]) -> None:
        intents_parent = self.__intents_client.project_agent_path(self.__project)

        # Group the training data by intents
        intent_sentences = {
            intent: [ datum for datum in training_data if datum.intent == intent ]
            for intent
            in { x.intent for x in training_data }
        }

        # Convert the training data into the format expected by Dialogflow
        # Each intent becomes an object, containing the training data for that specific intent
        intent_instances = [
            dialogflow_v2.types.Intent(
                display_name     = intent,
                ml_disabled      = False, # Explicitly don't disable machine learning
                training_phrases = [
                    dialogflow_v2.types.Intent.TrainingPhrase(
                        type  = dialogflow_v2.types.Intent.TrainingPhrase.Type.EXAMPLE,
                        parts = [ dialogflow_v2.types.Intent.TrainingPhrase.Part(
                            text=datum.sentence
                        ) ]
                    ) for datum in data
                ]
            ) for intent, data in intent_sentences.items()
        ]

        # Manually add a fallback intent to represent the None-intent
        intent_instances.append(dialogflow_v2.types.Intent(
            display_name = "None",
            is_fallback  = True
        ))

        intent_batch = dialogflow_v2.types.IntentBatch(intents=intent_instances)

        # Create the intents
        self.__intents_client.batch_update_intents(
            intents_parent,
            self.__language,
            intent_batch_inline=intent_batch
        ).result() # TODO: Replace .result() with a non-blocking alternative

        # Train the agent
        self.__agents_client.train_agent(self.__agents_client.project_path(self.__project)).result()

    async def rateIntents(self, sentence: str) -> NLUIntentRating:
        # The session id is randomized so that the context-mechanics of Dialogflow don't mess with
        # the result. The maximum length allowed by Dialogflow is 36 bytes.
        session_id = "".join(random.choices(string.ascii_letters + string.digits, k=36))
        session = self.__sessions_client.session_path(self.__project, session_id)

        text_input  = dialogflow_v2.types.TextInput(text=sentence, language_code=self.__language)
        query_input = dialogflow_v2.types.QueryInput(text=text_input)

        while True:
            try:
                detect_intent_response = self.__sessions_client.detect_intent(session, query_input)
                break
            except FailedPrecondition:
                # TODO: Remove this as soon as the problem described in
                # https://github.com/googleapis/dialogflow-python-client-v2/issues/171 is resolved.
                # Some race condition in the Dialogflow API causes this exception to be raised
                # nondeterministically. Waiting a bit usually makes the issue disappear.
                await asyncio.sleep(5)
            except ResourceExhausted:
                # The code managed to exceed the quota for text queries. Wait a bit and try again.
                await asyncio.sleep(10)

        intent = (
            None
            if detect_intent_response.query_result.intent.is_fallback else
            detect_intent_response.query_result.intent.display_name
        )

        return NLUIntentRating(sentence, [ (
            intent,
            detect_intent_response.query_result.intent_detection_confidence
        ) ])

    async def cleanupTraining(self) -> None:
        await self.__removeIntents()

    async def __removeIntents(self) -> None:
        """
        Remove all intents from the Dialogflow agent.
        """

        intents_parent = self.__intents_client.project_agent_path(self.__project)

        intents = list(self.__intents_client.list_intents(intents_parent))

        if len(intents) > 0:
            self.__intents_client.batch_delete_intents(intents_parent, intents).result()
