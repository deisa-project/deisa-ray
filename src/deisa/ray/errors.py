import traceback
import sys


class ContractError(Exception):
    """Exception raised when a contract or invariant is violated."""

    def __init__(self, message="Contract not satisfied."):
        """
        Initialize the contract error.

        Parameters
        ----------
        message : str, optional
            Human-readable explanation of the violated contract.
        """
        super().__init__(message)
        self.message = message


class ConfigError(RuntimeError):
    """Raised when configuration is mutated after it has been locked."""

    pass


def _default_exception_handler(e: BaseException):
    """
    Print an exception traceback to stderr.

    Parameters
    ----------
    e : BaseException
        Exception to report.

    Returns
    -------
    None
        The exception is reported for debugging and not re-raised.
    """
    traceback.print_exc(file=sys.stderr)
    print(e, file=sys.stderr)
