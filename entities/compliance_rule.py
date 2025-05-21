from typing import Callable, Any

class ComplianceRule:
    def __init__(self, name: str, description: str, validate: Callable[[Any], bool]):
        self.name = name
        self.description = description
        self.validate = validate 