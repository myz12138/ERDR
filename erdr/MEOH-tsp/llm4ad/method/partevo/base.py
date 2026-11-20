
from llm4ad.base.code import Function
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class Evoind:
    """
    Evolutionary Individual used within the PartEvo framework.
    Wraps a Function object and stores its metadata within a specific algorithm cluster.

    An Evoind is the basic unit for selection, elimination, and other operations
    within the algorithm pool.
    """
    function: Function
    cluster_id: Optional[int] = None

    # --- Evolutionary Metadata ---
    reflection: str = ""
    feature: List[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return str(self.function)

    def __hash__(self):
        """Implement hashing based on the underlying Function object for deduplication."""
        return hash(self.function)

    def __eq__(self, other):
        """Compare if two Individuals have the same underlying Function object."""
        if not isinstance(other, Evoind):
            return NotImplemented
        return self.function == other.function

    def set_feature(self, feature_given):
        self.feature = feature_given

    def set_reflection(self, reflection_given):
        self.reflection = reflection_given
