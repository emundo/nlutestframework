import asyncio
import math

from ..nlu_intent_rating import NLUIntentRating
from ..optimizable_nlu_framework import OptimizableNLUFramework

# API for LUIS
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials

# Language tag handling
from langcodes import Language

# Other imports only for the type hints
from typing import List, Optional
from ..global_config import GlobalConfig
from ..nlu_data_entry import NLUDataEntry
from ..nlu_data_set import NLUDataSet

class LUISNLUFramework(OptimizableNLUFramework):
    # LUIS requires assigning a version to each app. As this code doesn't require any versioning of
    # the app, a placeholder/fake version is used.
    FAKE_VERSION = "1.0"

    # pylint: disable=arguments-differ
    async def construct( # type: ignore
        self,
        _global_config: GlobalConfig,
        endpoint: str,
        authoring_key: str,
        *args,
        runtime_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Args:
            _global_config: Global configuration for the whole test framework.
            endpoint: The endpoint to use to access LUIS, e.g.
                `https://westeurope.api.cognitive.microsoft.com/`.
            authoring_key: The access key for the LUIS authoring API.
            runtime_key: The access key for the LUIS runtime API. Defaults to the authoring key if
                omitted or set to :obj:`None`.
        """

        await super().construct(*args, **kwargs)

        self.__authoring_client = LUISAuthoringClient(
            endpoint,
            CognitiveServicesCredentials(authoring_key)
        )

        self.__runtime_client = LUISRuntimeClient(
            endpoint,
            CognitiveServicesCredentials(authoring_key if runtime_key is None else runtime_key)
        )

    # pylint: disable=attribute-defined-outside-init
    async def _prepareDataSet(self, data_set: NLUDataSet) -> None:
        self.__app_id = self.__authoring_client.apps.add({
            "name"    : "NLUTestFramework",
            "culture" : Language.get(data_set.language).simplify_script().to_tag(),
            "initial_version_id": self.__class__.FAKE_VERSION
        })

    async def unprepareDataSet(self) -> None:
        self.__authoring_client.apps.delete(self.__app_id, force=True)

        del self.__app_id

    # pylint: disable=attribute-defined-outside-init
    async def train(self, training_data: List[NLUDataEntry]) -> None:
        fake_version = self.__class__.FAKE_VERSION

        self.__intent_ids = []

        for intent in { x.intent for x in training_data }:
            self.__intent_ids.append(self.__authoring_client.model.add_intent(
                self.__app_id,
                fake_version,
                intent
            ))

        # Prepare the examples
        examples = [ { "text": x.sentence, "intent_name": x.intent } for x in training_data ]

        # Add all examples, in batches of 100
        for i in range(0, math.ceil(len(examples) / 100)):
            batch = examples[i * 100:(i + 1) * 100]
            self.__authoring_client.examples.batch(self.__app_id, fake_version, batch)

        # Train the model
        self.__authoring_client.train.train_version(self.__app_id, fake_version)

        # Wait for the training to complete
        while True:
            # get_status returns a list of training statuses, one for each model.
            statuses = self.__authoring_client.train.get_status(self.__app_id, fake_version)

            unpacked_statuses = [ x.details.status for x in statuses ]

            # Check if any models are still training/queued for training
            if "Queued" in unpacked_statuses or "InProgress" in unpacked_statuses:
                await asyncio.sleep(1)
            else:
                # Get the list of models whose' training failed and the reasons of their failure
                failed_models = filter(lambda x: x.details.status == "Fail", statuses)
                failure_reasons = [ x.details.failure_reason for x in failed_models ]

                # Check if any of the models failed training
                if len(failure_reasons) > 0:
                    # TODO: Better formating of failure_reasons.
                    raise Exception("Training failed: {}".format(failure_reasons))

                break

        # Publish the trained app
        self.__authoring_client.apps.publish(self.__app_id, fake_version, is_staging=True)

    async def _rateIntents(self, sentence: str) -> NLUIntentRating:
        prediction = self.__runtime_client.prediction.get_slot_prediction(
            app_id             = self.__app_id,
            slot_name          = "staging",
            prediction_request = { "query": sentence }
        ).prediction.intents

        return NLUIntentRating(
            sentence,
            [ (intent, x.score) for intent, x in prediction.items() ]
        )

    async def cleanupTraining(self) -> None:
        # Delete all intents and the corresponding utterances
        for intent_id in self.__intent_ids:
            self.__authoring_client.model.delete_intent(
                self.__app_id,
                self.__class__.FAKE_VERSION,
                intent_id,
                delete_utterances=True
            )

        del self.__intent_ids
