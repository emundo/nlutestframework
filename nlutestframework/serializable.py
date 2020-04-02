# Other imports only for the type hints
from .types import JSONSerializable

class Serializable:
    def serialize(self) -> JSONSerializable:
        """
        Returns:
            A representation of this instance which consists only of primitive data types, lists and
            dictionaries. The value returned by this method is safe to be serialized to JSON and
            other formats with similar capabilities.
        """

        raise NotImplementedError

    @classmethod
    def fromSerialized(cls, serialized: JSONSerializable) -> "Serializable":
        """
        Constructs and returns an instance given a value as returned by :meth:`serialize`.

        Args:
            serialized: A representation of this instance which consists only of primitive data
                types, lists and dictionaries, as returned by :meth:`serialize`.

        Returns:
            A new instance filled with the data contained in the serialized value.
        """

        raise NotImplementedError
