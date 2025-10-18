from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from data import EdgeLabel, EdgeProbability, ReasoningStep

SemanticLLMCallable = Callable[[Sequence[ReasoningStep], float, int], str]


@dataclass(slots=True)
class SemanticSamplerConfig:
    """
    Configuration mirroring Algorithm 2 in Section 3.3.

    Parameters
    ----------
    max_samples: hard cap R_max on global adjacency samples.
    min_samples: minimum number of samples before convergence is tested.
    epsilon: confidence threshold for the pooled standard error.
    tau_positive / tau_negative: decision thresholds for signed confidence.
    temperatures: decoding temperature schedule τ_r used for LLM sampling.
    """

    max_samples: int = 16
    min_samples: int = 4
    epsilon: float = 0.05
    tau_positive: float = 0.4
    tau_negative: float = 0.3
    temperatures: Tuple[float, ...] = (0.3, 0.5, 0.7)

    def temperature_for(self, sample_idx: int) -> float:
        if not self.temperatures:
            return 0.0
        return self.temperatures[sample_idx % len(self.temperatures)]


@dataclass(slots=True)
class EdgeConfidence:
    probability: EdgeProbability
    samples: int
    standard_error: float

    def signed_confidence(self) -> float:
        return self.probability.support - self.probability.contradict


class SemanticSampler:
    """
    Adaptive LLM-driven sampler for semantic edge detection.
    """

    def __init__(self, llm_callable: SemanticLLMCallable, config: SemanticSamplerConfig | None = None) -> None:
        self.llm_callable = llm_callable
        self.config = config or SemanticSamplerConfig()

    def run(
        self,
        steps: Sequence[ReasoningStep],
    ) -> Tuple[Dict[Tuple[str, str], EdgeConfidence], Dict[Tuple[str, str], EdgeLabel]]:
        keys = [step.key for step in steps]
        pairs = _ordered_pairs(keys)

        counts: Dict[Tuple[str, str], Dict[EdgeLabel, int]] = {
            pair: {label: 0 for label in EdgeLabel} for pair in pairs
        }

        for sample_idx in range(self.config.max_samples):
            temperature = self.config.temperature_for(sample_idx)
            payload = self.llm_callable(steps, temperature, sample_idx)
            sample = parse_semantics_payload(payload)
            enriched = _complete_sample(sample, pairs)

            for pair, label in enriched.items():
                counts[pair][label] += 1

            if sample_idx + 1 >= self.config.min_samples:
                confidences = {
                    pair: _estimate_confidence(freqs)
                    for pair, freqs in counts.items()
                }
                if _converged(confidences, self.config.epsilon):
                    break

        confidences = {
            pair: _estimate_confidence(freqs)
            for pair, freqs in counts.items()
        }
        adjacency = _decide_edges(confidences, self.config.tau_positive, self.config.tau_negative)
        mirrored = _mirror_adjacency(adjacency, keys)

        return confidences, mirrored


def parse_semantics_payload(payload: str) -> Dict[Tuple[str, str], EdgeLabel]:
    data = _load_json(payload)
    result: Dict[Tuple[str, str], EdgeLabel] = {}
    for key, value in data.items():
        pair = _parse_pair_key(key)
        result[pair] = EdgeLabel.from_string(value)
    return result


def _ordered_pairs(keys: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i, src in enumerate(keys):
        for j in range(i + 1, len(keys)):
            pairs.append((src, keys[j]))
    return pairs


def _complete_sample(
    sample: Mapping[Tuple[str, str], EdgeLabel],
    pairs: Iterable[Tuple[str, str]],
) -> Dict[Tuple[str, str], EdgeLabel]:
    enriched: Dict[Tuple[str, str], EdgeLabel] = {}
    for pair in pairs:
        enriched[pair] = sample.get(pair, EdgeLabel.INDEPENDENT)
    return enriched


def _estimate_confidence(freqs: Mapping[EdgeLabel, int]) -> EdgeConfidence:
    total = sum(freqs.values())
    if total <= 0:
        probability = EdgeProbability(0.0, 0.0, 1.0)
        return EdgeConfidence(probability=probability, samples=0, standard_error=float("inf"))

    support = freqs[EdgeLabel.SUPPORT] / total
    contradict = freqs[EdgeLabel.CONTRADICT] / total
    independent = freqs[EdgeLabel.INDEPENDENT] / total
    probability = EdgeProbability(support, contradict, independent)

    se = math.sqrt(max(0.0, (support * (1 - support) + contradict * (1 - contradict)) / total))
    return EdgeConfidence(probability=probability, samples=total, standard_error=se)


def _converged(confidences: Mapping[Tuple[str, str], EdgeConfidence], epsilon: float) -> bool:
    if not confidences:
        return False
    max_se = max(conf.standard_error for conf in confidences.values())
    return max_se <= epsilon


def _decide_edges(
    confidences: Mapping[Tuple[str, str], EdgeConfidence],
    tau_pos: float,
    tau_neg: float,
) -> Dict[Tuple[str, str], EdgeLabel]:
    adjacency: Dict[Tuple[str, str], EdgeLabel] = {}
    for pair, confidence in confidences.items():
        signed = confidence.signed_confidence()
        if signed >= tau_pos:
            adjacency[pair] = EdgeLabel.SUPPORT
        elif signed <= -tau_neg:
            adjacency[pair] = EdgeLabel.CONTRADICT
        else:
            adjacency[pair] = EdgeLabel.INDEPENDENT
    return adjacency


def _mirror_adjacency(
    adjacency: Mapping[Tuple[str, str], EdgeLabel],
    keys: Sequence[str],
) -> Dict[Tuple[str, str], EdgeLabel]:
    mirrored: Dict[Tuple[str, str], EdgeLabel] = {}
    for i, src in enumerate(keys):
        for j, dst in enumerate(keys):
            if i == j:
                continue
            if i < j:
                label = adjacency.get((src, dst), EdgeLabel.INDEPENDENT)
            else:
                label = _flip_label(adjacency.get((dst, src), EdgeLabel.INDEPENDENT))
            mirrored[(src, dst)] = label
    return mirrored


def _parse_pair_key(key: str) -> Tuple[str, str]:
    stripped = key.strip().strip("()")
    left, right = stripped.split(",")
    return left.strip(), right.strip()


def _flip_label(label: EdgeLabel) -> EdgeLabel:
    if label == EdgeLabel.SUPPORT:
        return EdgeLabel.CONTRADICT
    if label == EdgeLabel.CONTRADICT:
        return EdgeLabel.SUPPORT
    return EdgeLabel.INDEPENDENT


def _load_json(payload: str) -> dict:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("prompt response does not contain a JSON object")
    return json.loads(payload[start : end + 1])
