from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from data import ReasoningStep, ReasoningUnit
from embedding import BaseEmbedder

ClusterLLMCallable = Callable[[Sequence[ReasoningUnit], float, int], str]


@dataclass(slots=True)
class ClusteringScore:
    """Scores derived from the three criteria defined in Section 3.2."""

    intra_coherence: float
    separation: float
    length_regularisation: float

    def weighted(self, weights: Tuple[float, float, float]) -> float:
        w_ic, w_sep, w_len = weights
        return (
            w_ic * self.intra_coherence
            + w_sep * self.separation
            + w_len * self.length_regularisation
        )


@dataclass(slots=True)
class ClusterCandidate:
    steps: List[ReasoningStep]
    temperature: float
    sample_index: int
    raw_json: str | None = None
    score: ClusteringScore | None = None


@dataclass(slots=True)
class ClusteringConfig:
    """
    Configuration for the LLM-based clustering ensemble.

    Parameters
    ----------
    num_samples: number of candidates B sampled from the LLM.
    temperatures: decoding temperature schedule τ_r used during sampling.
    weights: (w_ic, w_sep, w_len) weights for the three criteria, summing to 1.
    """

    num_samples: int = 8
    temperatures: Tuple[float, ...] = (0.3, 0.5, 0.7)
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)

    def temperature_for(self, sample_idx: int) -> float:
        if not self.temperatures:
            return 0.0
        return self.temperatures[sample_idx % len(self.temperatures)]

    def normalised_weights(self) -> Tuple[float, float, float]:
        total = sum(self.weights)
        if total <= 0.0:
            raise ValueError("clustering weights must sum to a positive value")
        return tuple(weight / total for weight in self.weights)


class ClusterSampler:
    """
    Invoke an LLM to obtain multiple clustering candidates following Template 9.
    """

    def __init__(self, llm_callable: ClusterLLMCallable, config: ClusteringConfig | None = None) -> None:
        self.llm_callable = llm_callable
        self.config = config or ClusteringConfig()

    def sample(self, units: Sequence[ReasoningUnit]) -> List[ClusterCandidate]:
        units = list(units)
        if not units:
            raise ValueError("cannot cluster an empty set of reasoning units")

        candidates: List[ClusterCandidate] = []
        unit_count = len(units)

        for sample_idx in range(self.config.num_samples):
            temperature = self.config.temperature_for(sample_idx)
            payload = self.llm_callable(units, temperature, sample_idx)
            steps = parse_cluster_json(payload)
            _validate_unit_indices(steps, unit_count)
            candidates.append(
                ClusterCandidate(
                    steps=steps,
                    temperature=temperature,
                    sample_index=sample_idx,
                    raw_json=payload,
                )
            )

        return candidates


class ClusterSelector:
    """
    Score the ensemble of candidates using the criteria defined in Section 3.2.
    """

    def __init__(self, embedder: BaseEmbedder, config: ClusteringConfig | None = None) -> None:
        self.embedder = embedder
        self.config = config or ClusteringConfig()

    def select(self, units: Sequence[ReasoningUnit], candidates: Iterable[ClusterCandidate]) -> ClusterCandidate:
        units = list(units)
        if not units:
            raise ValueError("no reasoning units provided")

        unit_vectors = normalise_vectors(self.embedder.embed([unit.text for unit in units]))
        if len(unit_vectors) != len(units):
            raise ValueError("embedder returned unexpected shape for reasoning units")

        mu_ref = reference_step_length(len(units))
        weights = self.config.normalised_weights()

        best_candidate: ClusterCandidate | None = None
        best_score = float("-inf")

        for candidate in candidates:
            score = self._score_candidate(candidate.steps, unit_vectors, mu_ref)
            candidate.score = score
            composite = score.weighted(weights)
            if composite > best_score:
                best_score = composite
                best_candidate = candidate

        if best_candidate is None:
            raise ValueError("LLM did not yield any valid clustering candidates")
        return best_candidate

    def _score_candidate(
        self,
        steps: Sequence[ReasoningStep],
        unit_vectors: Sequence[Sequence[float]],
        mu_ref: float,
    ) -> ClusteringScore:
        if not steps:
            return ClusteringScore(0.0, 0.0, 0.0)

        intra_scores: List[float] = []
        for step in steps:
            indices = list(step.unit_indices)
            if len(indices) <= 1:
                intra_scores.append(1.0)
                continue
            pairwise: List[float] = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairwise.append(cosine_similarity(unit_vectors[indices[i]], unit_vectors[indices[j]]))
            intra_scores.append(statistics.fmean(pairwise) if pairwise else 1.0)

        intra_coherence = statistics.fmean(intra_scores) if intra_scores else 0.0

        step_vectors = normalise_vectors(self.embedder.embed([step.content for step in steps]))
        if len(step_vectors) != len(steps):
            raise ValueError("embedder returned unexpected shape for reasoning steps")

        separation_samples: List[float] = []
        for idx in range(len(steps) - 1):
            cosine = cosine_similarity(step_vectors[idx], step_vectors[idx + 1])
            separation_samples.append(1.0 - cosine)
        separation = statistics.fmean(separation_samples) if separation_samples else 1.0

        avg_length = float(sum(len(step.unit_indices) for step in steps)) / len(steps)
        length_regularisation = 1.0 if mu_ref <= 0 else 1.0 - abs(avg_length / mu_ref - 1.0)

        return ClusteringScore(
            intra_coherence=intra_coherence,
            separation=separation,
            length_regularisation=length_regularisation,
        )


def parse_cluster_json(payload: str) -> List[ReasoningStep]:
    """
    Parse the JSON response produced by Template 9 (Pclu).
    """

    data = _load_json(payload)
    steps: List[ReasoningStep] = []

    for key in sorted(data.keys(), key=_sort_key):
        entry = data[key]
        title = entry.get("title", "").strip() or key
        content = entry.get("content", "").strip()
        raw_indices = entry.get("unit_indices")
        if raw_indices is None:
            raise ValueError(f"missing `unit_indices` for step {key}")
        indices = tuple(int(idx) for idx in raw_indices)
        steps.append(
            ReasoningStep(
                key=key,
                title=title,
                content=content,
                unit_indices=indices,
            )
        )
    return steps


def reference_step_length(num_units: int) -> float:
    if num_units <= 0:
        return 1.0
    k_target = max(3, math.ceil(math.sqrt(num_units)))
    k_target = min(k_target, 30, num_units)
    return max(1.0, num_units / k_target)


def normalise_vectors(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    normalised: List[List[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0.0:
            normalised.append([0.0 for _ in vector])
        else:
            normalised.append([component / norm for component in vector])
    return normalised


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0
    dot = sum(vec_a[i] * vec_b[i] for i in range(length))
    return max(-1.0, min(1.0, dot))


def _validate_unit_indices(steps: Sequence[ReasoningStep], unit_count: int) -> None:
    valid_indices = set(range(unit_count))
    for step in steps:
        if not step.unit_indices:
            raise ValueError("each reasoning step must reference at least one reasoning unit")
        for index in step.unit_indices:
            if index not in valid_indices:
                raise ValueError(f"reasoning step references unknown unit index {index}")


def _sort_key(step_key: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in step_key if ch.isdigit())
    if digits:
        try:
            return (int(digits), step_key)
        except ValueError:
            pass
    return (0, step_key)


def _load_json(payload: str) -> dict:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("prompt response does not contain a JSON object")
    return json.loads(payload[start : end + 1])
