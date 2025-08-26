from typing import Set
class FormalConcept:
    def __init__(self, intent: int, extent: int):
        self.extent: int = extent
        self.intent: int = intent