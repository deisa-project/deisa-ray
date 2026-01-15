class ContractError(Exception):
    """Exception raised when a contract or invariant is violated."""

    def __init__(self, message="Contract not satisfied."):
        super().__init__(message)
        self.message = message


class ConfigError(RuntimeError):
    """Raised when configuration is mutated after it has been locked."""

    pass

def _default_exception_handler(e: BaseException):
    print(f"There was an error {e} in the callback! Unregistering it!")
    

