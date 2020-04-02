import json

from ..nlu_data_entry import NLUDataEntry
from ..nlu_data_set import NLUDataSet

# Other imports only for the type hints
from typing import List

class SimpleJSONDataSet(NLUDataSet):
    def _loadData(self, data_path: str) -> List[NLUDataEntry]:
        with open(data_path, "r") as f:
            data = json.load(f)

        # Load the language from the data set
        self._setLanguage(data["lang"])

        # Load the intents and sentences from the data set
        return [ NLUDataEntry(
            entry["text"],
            None if entry["intent"] == "None" else entry["intent"]
        ) for entry in data["sentences"] ]
