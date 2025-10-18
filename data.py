from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReasoningUnit:
    """Minimal segment produced after delimiter-based splitting."""

    index: int
    text: str


@dataclass(frozen=True)
class ReasoningStep:
    """
    Semantically coherent reasoning step used as a node in the reasoning graph.
    """

    key: str
    title: str
    content: str
    unit_indices: Tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, str]:
        return {"title": self.title, "content": self.content}


class EdgeLabel(str, Enum):
    """Discrete semantic relation between two reasoning steps."""

    SUPPORT = "support"
    CONTRADICT = "contradict"
    INDEPENDENT = "independent"

    @classmethod
    def from_string(cls, value: str) -> "EdgeLabel":
        lower = value.strip().lower()
        for member in cls:
            if member.value == lower:
                return member
        raise ValueError(f"unknown edge label: {value!r}")


@dataclass(frozen=True)
class EdgeProbability:
    """Estimated probabilities for each semantic relation."""

    support: float
    contradict: float
    independent: float

    def normalized(self) -> "EdgeProbability":
        total = self.support + self.contradict + self.independent
        if total <= 0:
            return EdgeProbability(0.0, 0.0, 1.0)
        return EdgeProbability(
            support=self.support / total,
            contradict=self.contradict / total,
            independent=self.independent / total,
        )

    def signed_confidence(self) -> float:
        """Signed confidence score w_ij = p(+1) - p(-1)."""
        return self.support - self.contradict


@dataclass
class ReasoningGraph:
    """
    Directed reasoning graph.

    Attributes
    ----------
    steps: Ordered list of reasoning steps.
    adjacency: Maps (src_key, dst_key) to a discrete edge label.
    weights: Optional signed confidence for the same edge.
    """

    steps: Sequence[ReasoningStep]
    adjacency: Dict[Tuple[str, str], EdgeLabel] = field(default_factory=dict)
    weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def ordered_keys(self) -> List[str]:
        return [step.key for step in self.steps]

    def iter_edges(self) -> Iterable[Tuple[str, str, EdgeLabel]]:
        for (src, dst), label in self.adjacency.items():
            yield src, dst, label

    def get_weight(self, src: str, dst: str) -> Optional[float]:
        return self.weights.get((src, dst))

    def to_networkx(self):  # pragma: no cover - optional helper
        import networkx as nx

        graph = nx.DiGraph()
        for step in self.steps:
            graph.add_node(step.key, title=step.title, content=step.content)
        for (src, dst), label in self.adjacency.items():
            graph.add_edge(src, dst, label=label.value, weight=self.weights.get((src, dst)))
        return graph
