"""Exceptions raised by SmartPool."""


class SmartPoolError(Exception):
    """Base class for SmartPool errors."""


class WorkerLostError(SmartPoolError):
    """A worker's executor exited while the task was still running.

    Raised instead of leaving the future pending forever when a child process
    dies mid-task, for example after a segfault, an out-of-memory kill, or a
    failure to unpickle the task payload.
    """


__all__ = ["SmartPoolError", "WorkerLostError"]