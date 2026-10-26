"""User-facing error types. The CLI prints these without a traceback (unless --debug)."""


class VisionBenchError(Exception):
    """Base class for expected errors caused by input or state, not by bugs."""


class DatasetValidationError(VisionBenchError):
    pass


class DatasetExistsError(VisionBenchError):
    pass


class DatasetNotFoundError(VisionBenchError):
    pass


class SplitError(VisionBenchError):
    pass
