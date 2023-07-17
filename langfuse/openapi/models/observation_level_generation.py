from enum import Enum


class ObservationLevelGeneration(str, Enum):
    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    ERROR = "ERROR"
    WARNING = "WARNING"

    def __str__(self) -> str:
        return str(self.value)
