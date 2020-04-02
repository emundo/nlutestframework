import asyncio

# Other imports only for the type hints
from typing import List, Tuple, Any, Callable, Awaitable, Optional, TypeVar

O = TypeVar("O") # pylint: disable=invalid-name
R = TypeVar("R") # pylint: disable=invalid-name

class ParallelException(Exception):
    """
    Exception thrown by :func:`~nlutestframework.parallel_exception.run_in_parallel`. Contains
    information about exceptions thrown during the operation and the rollback phases.
    """

    def __init__(
        self,
        message: str,
        operation_exceptions: List[Tuple[BaseException, O]],
        rollback_exceptions: List[Tuple[BaseException, O, R]]
    ):
        """
        Args:
            message: The exception message.
            operation_exceptions: Information about problems during the operation phase. The second
                entry denotes the object.
            rollback_exceptions: Information about problems during the rollback phase. The second
                entry denotes the object and the third entry denotes the result of the operation
                phase for that object.
        """

        super().__init__(message)

        self.__operation_exceptions = operation_exceptions
        self.__rollback_exceptions  = rollback_exceptions

    @property
    def operation_exceptions(self) -> List[Tuple[BaseException, O]]:
        """
        Returns:
            A copy of the list passed to the constructor, not a reference to it.
        """

        return list(self.__operation_exceptions)

    @property
    def rollback_exceptions(self) -> List[Tuple[BaseException, O, R]]:
        """
        Returns:
            A copy of the list passed to the constructor, not a reference to it.
        """

        return list(self.__rollback_exceptions)

    def __str__(self) -> str:
        operation_exception_messages = ""
        for exc, obj in self.__operation_exceptions:
            operation_exception_messages += "\tObject {}: {}\n".format(obj, exc)

        rollback_exception_messages = ""
        for exc, obj, res in self.__rollback_exceptions:
            rollback_exception_messages += "\tObject {}; Result {}: {}\n".format(obj, res, exc)

        return (
              "Exceptions during the operations phase:\n"
            + operation_exception_messages
            + "Exceptions during the rollback phase:\n"
            + rollback_exception_messages
        )

# pylint thinks "op" is too short
# pylint: disable=invalid-name
async def run_in_parallel(
    objects: List[O],
    op: Callable[[O], Awaitable[R]],
    rollback_op: Optional[Callable[[O, R], Awaitable[Any]]],
    message: str
) -> List[R]:
    """
    Run the same asynchronous operation on multiple objects in parallel. If the operation fails for
    at least one of the objects, run a rollback operation on the objects that succeeded in the
    operation. This can be used to implement an "all-or-nothing" mechanism. Either all operations
    succeed, or the succeeded operations are rolled back.

    In case of a failure followed by a rollback, a
    :exc:`~nlutestframework.parallel_exception.ParallelException` is raised which contains
    information about the problems that led to the rollback and information about problems during
    the rollback.

    Args:
        objects: The objects to run the operation on.
        op: The operation to run on each object. The object is passed as the first parameter.
        rollback_op: A function which is run once for each object which succeeded in the operation
            in case at least one operation failed. Receives two parameters: the result of the
            operation and the object. Pass :obj:`None` to skip the rollback-phase.
        message: The message to set on the exception that is thrown in case at least one operation
            failed and the whole set of operations was rolled back.

    Returns:
        A (future, which resolves to a) list containing the result of the operation applied to each
        object.

    Raises:
        :exc:`ParallelException`: If at least one operation failed and the whole set of operations
            was rolled back.
    """

    # Run the operation on each object
    results = await asyncio.gather(*map(op, objects), return_exceptions=True)

    # Split the pairs into successful and failed
    succeeded = list(filter(lambda x: not isinstance(x[1], BaseException), zip(objects, results)))
    failed    = list(filter(lambda x:     isinstance(x[0], BaseException), zip(results, objects)))

    if len(failed) == 0:
        # If there are no failed objects, return the results
        return results

    rollback_failures = []

    if rollback_op is not None:
        # Run the rollback operation on the objects that succeeded
        results = await asyncio.gather(
            # mypy ignores the previous check for rollback_op to not be None
            *map(lambda x: rollback_op(*x), succeeded), # type: ignore
            return_exceptions=True
        )

        # Unpack the values into triples: (rollback_result, object, operation_result)
        results = [ (x[0], x[1][0], x[1][1]) for x in zip(results, succeeded) ]

        # Extract only the failed rollbacks
        rollback_failures = list(filter(lambda x: isinstance(x[0], BaseException), results))

    # Raise an exception containing information about all problems during the operation and the
    # rollback phases.
    raise ParallelException(message, failed, rollback_failures)
