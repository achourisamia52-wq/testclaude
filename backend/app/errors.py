class UpstreamError(Exception):
    """A provider (speech-to-text or Claude) failed. Maps to an HTTP status."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
