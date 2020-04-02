import asyncio

import docker
from langcodes import Language
import requests
import yaml

from ..nlu_intent_rating import NLUIntentRating
from ..optimizable_nlu_framework import OptimizableNLUFramework

# Other imports only for the type hints
from typing import List
import logging
from ..global_config import GlobalConfig
from ..nlu_data_entry import NLUDataEntry
from ..nlu_data_set import NLUDataSet

def _raise_for_status(response: requests.Response, logger: logging.Logger) -> requests.Response:
    """
    Check the status code of the response. If the status code suggests an error, log the response as
    JSON and then raise an exception. Otherwise, return the response for further processing.

    Args:
        response: The response to an HTTP request by the "requests" library.
        logger: The logger to log errors with.

    Returns:
        The response, in case it didn't fail.

    Raises:
        :exc:`requests.exceptions.HTTPError`: if the response indicates an error.
    """

    try:
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(response.json(), exc_info=e)
        raise

class RasaNLUFramework(OptimizableNLUFramework):
    __VERSION = "latest"

    # pylint: disable=arguments-differ
    async def construct( # type: ignore
        self,
        _global_config: GlobalConfig,
        pipeline: str,
        *args,
        timeout: int = 10,
        **kwargs
    ) -> None:
        """
        Args:
            _global_config: Global configuration for the whole test framework.
            pipeline: The pipeline to use by Rasa NLU. Must be either "supervised" or "pretrained".
                See https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/ for details.
            timeout: The time in seconds to wait for the Rasa HTTP server to start. Defaults to 10.
        """

        await super().construct(*args, **kwargs)

        if pipeline not in [ "supervised", "pretrained" ]:
            raise ValueError("The pipeline must be specified as either supervised or pretrained.")

        # TODO: Pipeline flexible aka based on the number of training samples?
        self.__pipeline = pipeline
        self.__timeout  = timeout

        self._logger.debug("Creating a client for the Docker daemon...")
        self.__docker = docker.from_env()

    # pylint: disable=attribute-defined-outside-init
    async def _prepareDataSet(self, data_set: NLUDataSet) -> None:
        language = Language.get(data_set.language).language

        if self.__pipeline == "supervised":
            pipeline_config = "supervised_embeddings"
            image = "rasa/rasa:{}".format(self.__VERSION)
        if self.__pipeline == "pretrained":
            pipeline_config = "pretrained_embeddings_spacy"
            # In theory it should be enough to install rasa/rasa:latest-spacy-{language}, but in
            # practice the training fails in these images due to the spaCy models not being found.
            # This bug is reported in the Rasa repo: https://github.com/RasaHQ/rasa/issues/4789
            image = "rasa/rasa:{}-spacy-{}".format(self.__VERSION, language)

        # Create the Rasa config
        self.__rasa_config_yml = yaml.dump({ "language": language, "pipeline": pipeline_config })

        # Connect to the Docker daemon and pull the Rasa container
        self._logger.info("Preparing the docker container for Rasa...")
        self._logger.debug("Pulling Rasa image \"%s\"...", image)
        self.__docker.images.pull(image)

        self._logger.debug("Starting the Rasa HTTP server...")
        self.__container = self.__docker.containers.run(
            image,

            # Run the Rasa server and enable the HTTP API
            [ "run", "--enable-api" ],

            # Automatically remove the container after the server shuts down
            auto_remove=True,

            # Don't wait for the command to finish
            detach=True,

            # Expose port 5005 (used for HTTP by Rasa) for TCP traffic to a random port
            ports={ "5005/tcp": None }
        )

        # Update the container information from the Docker daemon
        self.__container.reload()

        # Extract the port mapping and build the base url for the HTTP API
        port_mapping = self.__container.attrs["NetworkSettings"]["Ports"]["5005/tcp"][0]
        self.__url = "http://{}:{}/".format(port_mapping["HostIp"], port_mapping["HostPort"])

        self._logger.debug("Waiting for the health endpoint to come alive...")
        for _ in range(self.__timeout):
            try:
                success = requests.get(self.__url).status_code == 200
            except requests.exceptions.ConnectionError:
                success = False

            if success:
                break

            await asyncio.sleep(1)

        self._logger.info("Container running.")

    async def unprepareDataSet(self) -> None:
        self.__container.stop()

        del self.__rasa_config_yml
        del self.__container
        del self.__url

    # pylint: disable=attribute-defined-outside-init
    async def train(self, training_data: List[NLUDataEntry]) -> None:
        # Build the training data structure as required by Rasa
        training_markdown = ""

        for intent in { x.intent for x in training_data }:
            training_markdown += "## intent:{}\n".format(intent)

            for entry in filter(
                lambda x, intent=intent: x.intent == intent, # type: ignore
            training_data):
                training_markdown += "- {}\n".format(entry.sentence)

            training_markdown += "\n"

        self._logger.debug("Training a model...")
        training_response = _raise_for_status(requests.post(self.__url + "model/train", json={
            "config" : self.__rasa_config_yml,
            "nlu"    : training_markdown,
            "force"  : True,
            "save_to_default_model_directory": True
        }), self._logger)

        file_name = training_response.headers["filename"]
        self._logger.debug("Model file name: %s", file_name)

        self._logger.debug("Selecting the newly trained model...")
        _raise_for_status(requests.put(self.__url + "model", json={
            "model_file": "models/{}".format(file_name)
        }), self._logger)

        self._logger.debug("Training completed.")

    async def _rateIntents(self, sentence: str) -> NLUIntentRating:
        response = _raise_for_status(requests.post(
            self.__url + "model/parse",
            json={ "text": sentence }
        ), self._logger).json()

        return NLUIntentRating(sentence, [ (
            rated_intent["name"],
            rated_intent["confidence"]
        ) for rated_intent in response["intent_ranking"] ])

    async def cleanupTraining(self) -> None:
        _raise_for_status(requests.delete(self.__url + "model"), self._logger)
