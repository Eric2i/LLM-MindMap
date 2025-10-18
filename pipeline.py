from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from clustering import (
    ClusterCandidate,
    ClusteringConfig,
    ClusterLLMCallable,
    ClusterSampler,
    ClusterSelector,
)
from data import EdgeLabel, ReasoningGraph, ReasoningStep, ReasoningUnit
from embedding import BaseEmbedder
from graph import GraphMetrics, compute_metrics
from semantics import (
    EdgeConfidence,
    SemanticLLMCallable,
    SemanticSampler,
    SemanticSamplerConfig,
)


def split_reasoning_units(text: str) -> List[ReasoningUnit]:
    """
    Split raw Chain-of-Thought text using ``\\n\\n`` as the natural delimiter.
    """

    parts = [segment.strip() for segment in text.split("\n\n")]
    units: List[ReasoningUnit] = []
    idx = 0
    for part in parts:
        if not part:
            continue
        units.append(ReasoningUnit(index=idx, text=part))
        idx += 1
    return units


@dataclass(slots=True)
class PipelineConfig:
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    semantics: SemanticSamplerConfig = field(default_factory=SemanticSamplerConfig)


@dataclass(slots=True)
class PipelineResult:
    graph: ReasoningGraph
    metrics: GraphMetrics
    clustering: ClusterCandidate
    edge_confidence: Dict[Tuple[str, str], EdgeConfidence]


class MindMapPipeline:
    """
    High-level pipeline that implements the full reasoning-graph workflow.
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        cluster_llm: ClusterLLMCallable,
        semantic_llm: SemanticLLMCallable,
        config: PipelineConfig | None = None,
    ) -> None:
        cfg = config or PipelineConfig()
        self.embedder = embedder
        self.cluster_sampler = ClusterSampler(cluster_llm, cfg.clustering)
        self.cluster_selector = ClusterSelector(embedder=self.embedder, config=cfg.clustering)
        self.semantic_config = cfg.semantics
        self.semantic_llm = semantic_llm

    def run(
        self,
        reasoning_text: str,
        *,
        cluster_json_samples: Sequence[str] = (),
        semantic_json_samples: Sequence[str] = (),
    ) -> PipelineResult:
        units = split_reasoning_units(reasoning_text)
        if not units:
            raise ValueError("reasoning text did not yield any units")

        candidates = self._collect_cluster_candidates(units, cluster_json_samples)
        if not candidates:
            raise ValueError("no clustering candidates available")

        selected = self.cluster_selector.select(units, candidates)

        steps = selected.steps
        if not steps:
            raise ValueError("selected clustering returned no steps")

        edge_confidence, adjacency = self._run_semantics(steps, semantic_json_samples)

        graph = ReasoningGraph(steps=steps, adjacency=adjacency)
        metrics = compute_metrics(graph)
        return PipelineResult(
            graph=graph,
            metrics=metrics,
            clustering=selected,
            edge_confidence=edge_confidence,
        )

    def _collect_cluster_candidates(
        self,
        units: Sequence[ReasoningUnit],
        json_samples: Sequence[str],
    ) -> List[ClusterCandidate]:
        candidates = self.cluster_sampler.sample(units)
        base_index = -1
        for payload in json_samples:
            steps = self._parse_cluster_payload(payload)
            candidates.append(
                ClusterCandidate(
                    steps=steps,
                    temperature=0.0,
                    sample_index=base_index,
                    raw_json=payload,
                )
            )
            base_index -= 1
        return candidates

    def _run_semantics(
        self,
        steps: Sequence[ReasoningStep],
        json_samples: Sequence[str],
    ):
        llm_callable = self._semantic_llm_for(json_samples)
        sampler = SemanticSampler(llm_callable, self.semantic_config)
        return sampler.run(steps)

    def _parse_cluster_payload(self, payload: str) -> List[ReasoningStep]:
        from clustering import parse_cluster_json

        steps = parse_cluster_json(payload)
        if any(not step.unit_indices for step in steps):
            raise ValueError(
                "cluster JSON must include `unit_indices` for each step to align with raw units"
            )
        return steps

    def _parse_semantics_payload(self, payload: str) -> Dict[Tuple[str, str], EdgeLabel]:
        from semantics import parse_semantics_payload

        return parse_semantics_payload(payload)

    def _semantic_llm_for(self, payloads: Sequence[str]) -> SemanticLLMCallable:
        if not payloads:
            return self.semantic_llm

        cached = list(payloads)
        if not cached:
            return self.semantic_llm

        iterator = iter(cached)

        def replay_llm(
            _: Sequence[ReasoningStep],
            __: float,
            ___: int,
        ) -> str:
            try:
                return next(iterator)
            except StopIteration:
                return cached[-1]

        return replay_llm
