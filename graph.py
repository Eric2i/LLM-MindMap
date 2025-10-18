from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from data import EdgeLabel, ReasoningGraph


@dataclass(slots=True)
class GraphMetrics:
    num_steps: int
    num_edges: int
    exploration_density: float
    branching_ratio: float
    convergence_ratio: float
    linearity: float


def compute_metrics(graph: ReasoningGraph) -> GraphMetrics:
    steps = list(graph.steps)
    num_steps = len(steps)
    if num_steps == 0:
        return GraphMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)

    keys = [step.key for step in steps]
    adjacency = graph.adjacency

    directed_edges: Dict[Tuple[str, str], EdgeLabel] = {
        (src, dst): label
        for (src, dst), label in adjacency.items()
        if label != EdgeLabel.INDEPENDENT and src != dst
    }

    num_edges = len(directed_edges)
    possible_edges = max(1, num_steps * (num_steps - 1))
    exploration_density = num_edges / possible_edges

    out_degree = {key: 0 for key in keys}
    in_degree = {key: 0 for key in keys}

    for (src, dst), label in directed_edges.items():
        if src not in out_degree or dst not in in_degree:
            continue
        out_degree[src] += 1
        in_degree[dst] += 1

    branching_nodes = sum(1 for value in out_degree.values() if value > 1)
    convergence_nodes = sum(1 for value in in_degree.values() if value > 1)

    total_degree = {key: out_degree[key] + in_degree[key] for key in keys}
    nonlinear_nodes = sum(1 for value in total_degree.values() if value > 2)

    branching_ratio = branching_nodes / num_steps
    convergence_ratio = convergence_nodes / num_steps
    linearity = 1.0 - nonlinear_nodes / num_steps

    return GraphMetrics(
        num_steps=num_steps,
        num_edges=num_edges,
        exploration_density=exploration_density,
        branching_ratio=branching_ratio,
        convergence_ratio=convergence_ratio,
        linearity=linearity,
    )
