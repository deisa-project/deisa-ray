class ContractError(Exception):
    """Exception raised when a contract or invariant is violated."""

    def __init__(self, message="Contract not satisfied."):
        super().__init__(message)
        self.message = message
