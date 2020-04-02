import os

from nlutestframework.implementations import SimpleJSONDataSet

from langcodes import Language

script_directory  = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
corpora_directory = os.path.abspath(os.path.join(script_directory, "..", "data", "corpora"))

def runSimpleJSONDataSetTests(path, title, constructor_language, expected_language, size):
    expected_language = Language.get(expected_language).maximize().to_tag()

    # Run the tests twice, ignoring existing caches on the first run
    for ignore_cache in [ True, False ]:
        # Construct the data set
        data_set = SimpleJSONDataSet(title, path, 50, constructor_language, ignore_cache)

        assert data_set.title == title
        assert data_set.language == expected_language

        # Verify that the training data does not contain any None-intent sentences
        assert len(list(filter(lambda x: x.intent is None, data_set.training_data))) == 0

        # Get the number of None-intent sentences in the validation data
        num_none_intent_sentences = len(list(filter(
            lambda x: x.intent is None,
            data_set.validation_data
        )))

        # Verify that the training and validation data (without None-intent sentences) was split
        # correctly at about 50%
        validation_size_without_none = len(data_set.validation_data) - num_none_intent_sentences
        assert abs(len(data_set.training_data) - validation_size_without_none) <= 1

        # Verify that all entries were loaded
        assert len(data_set.training_data) + len(data_set.validation_data) == size

        # Make sure that the data returned on subsequent calls is the same
        assert data_set.training_data == data_set.training_data
        assert data_set.validation_data == data_set.validation_data

        # Verify that the data is sorted and split differently after reshuffling the data
        training_data = data_set.training_data
        validation_data = data_set.validation_data
        data_set.reshuffle()
        assert training_data != data_set.training_data
        assert validation_data != data_set.validation_data

        # Make sure that a copy of the data is returned and not a reference
        data_set.training_data.pop()
        data_set.validation_data.pop()
        assert len(data_set.training_data) + len(data_set.validation_data) == size

def test_AskUbuntuDataSet():
    runSimpleJSONDataSetTests(os.path.join(corpora_directory, "AskUbuntuCorpus.json"), "AskUbuntuCorpus", None, "en", 162)

def test_ChatbotDataSet():
    runSimpleJSONDataSetTests(os.path.join(corpora_directory, "ChatbotCorpus.json"), "ChatbotCorpus", None, "en", 206)

def test_WebApplicationsDataSet():
    runSimpleJSONDataSetTests(os.path.join(corpora_directory, "WebApplicationsCorpus.json"), "WebApplicationsCorpus", None, "en", 89)
