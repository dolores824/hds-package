"""
This class is used to raise an error to communicate problems to users in a friendly way.
"""

class DiaCcsPredError(Exception):
    """This is the error that should be thrown to help communicate problems to users in a nice way."""

    def __init__(self, message):
        """Instantiate the error with a given message"""
        self.message = message

    def __str__(self):
        return repr(self.message)