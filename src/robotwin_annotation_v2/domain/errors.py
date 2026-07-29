"""Domain-level errors."""


class DomainError(Exception):
    """Base for all domain errors."""
    pass


class InvalidStateTransition(DomainError):
    """Attempted an invalid state transition."""
    pass


class InvalidRequest(DomainError):
    """Request violates domain constraints."""
    pass


class ApprovalRequired(DomainError):
    """Operation requires human approval."""
    pass
