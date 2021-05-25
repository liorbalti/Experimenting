class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class FileLengthError(Error):
    """Exception raised when the file length is not appropriate

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, length, expected_length, message):
        self.length = length
        self.expected_length = expected_length
        self.message = message
