import logging

# Other imports only for the type hints
from typing import Optional

class HasLogger:
    """
    Base class for classes that want to make use of the Python :mod:`logging`-library.
    """

    def __init__(self, logger_title: Optional[str] = None):
        """
        Instantiate a :class:`~logging.Logger` with given title.

        Args:
            logger_title: The title of the logger to create for this instance. Defaults to the class
                name if omitted or set to :obj:`None`.
        """

        self._logger_title = self.__class__.__name__ if logger_title is None else logger_title

    @property
    def _logger(self) -> logging.Logger:
        """
        Returns:
            A logger instance with the title set to the value chosen during construction.
        """

        return logging.getLogger(self._logger_title)
