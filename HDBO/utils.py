class OptimizeException(Exception):
    def __init__(self, message) -> None:
        self.message = message
    def __str__(self) -> str:
        return self.message

class GradZerosException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
