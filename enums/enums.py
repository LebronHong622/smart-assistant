from enum import Enum

class OverflowMemoryMethod(Enum):
    TRIM = "trim"
    SUMMARY = "summary"
    DELETE = "delete"