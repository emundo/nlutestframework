class GlobalConfig:
    """
    Global configuration of the NLU test framework.
    """

    def __init__(self, python: str, iterations: int, ignore_cache: bool):
        """
        Args:
            python: The absolute path to a python executable. This executable can be used by
                :class:`~nlutestframework.nlu_framework.NLUFramework` implementations to call
                respective external modules. See the
                :class:`~nlutestframework.implementations.snips_nlu_framework.SnipsNLUFramework`
                implementation for an example.
            iterations: The number of iterations to measure the performances of the frameworks.
            ignore_cache: A boolean indicating whether to ignore cached data.
        """

        self.__python = python
        self.__iterations = iterations
        self.__ignore_cache = ignore_cache

    @property
    def python(self) -> str:
        return self.__python

    @property
    def iterations(self) -> int:
        return self.__iterations

    @property
    def ignore_cache(self) -> bool:
        return self.__ignore_cache
