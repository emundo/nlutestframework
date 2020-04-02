from .serializable import Serializable

# Other imports only for the type hints
from typing import TypeVar, Type
from .types import Intent, JSONSerializable

T = TypeVar("T", bound="NLUDataEntry")

class NLUDataEntry(Serializable):
    def __init__(self, sentence: str, intent: Intent):
        """
        Args:
            sentence: The sentence rated by the NLU framework.
            intent: The correct intent of the sentence.
        """

        self.__sentence = sentence
        self.__intent   = intent

    @property
    def sentence(self) -> str:
        return self.__sentence

    @property
    def intent(self) -> Intent:
        return self.__intent

    def serialize(self) -> JSONSerializable:
        return {
            "sentence" : self.__sentence,
            "intent"   : self.__intent
        }

    @classmethod
    def fromSerialized(cls: Type[T], serialized: JSONSerializable) -> T:
        return cls(serialized["sentence"], serialized["intent"]) # type: ignore

    def __str__(self) -> str:
        return "Intent for sentence \"{}\": \"{}\"".format(self.__sentence, self.__intent)
