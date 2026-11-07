"""Exceptions raised by the package.

The UI catches `GpsAnalysisError` and shows its message to the user, so the
messages are written for people, not for developers.
"""


class GpsAnalysisError(Exception):
    """Base class for errors the user can act on (bad input, bad settings)."""


class InputFormatError(GpsAnalysisError, ValueError):
    """An uploaded file cannot be read or is missing required fields."""


class InsufficientDataError(GpsAnalysisError, ValueError):
    """Not enough valid records to build a trajectory or run an analysis."""


class ConfigError(GpsAnalysisError, ValueError):
    """A configuration value is missing or invalid."""


class DatabaseError(GpsAnalysisError):
    """The optional PostGIS database is not configured or not reachable."""
