import traceback
import sys


class ContractError(Exception):
    """Exception raised when a contract or invariant is violated."""

    def __init__(self, message="Contract not satisfied."):
        super().__init__(message)
        self.message = message


class ConfigError(RuntimeError):
    """Raised when configuration is mutated after it has been locked."""

    pass


def _default_exception_handler(e: BaseException):
    """
    Print the traceback of an exception to stderr for debugging.

    Parameters
    ----------
    e : BaseException
        Exception to report.
    """
    traceback.print_exc(file=sys.stderr)
    print(e, file=sys.stderr)
