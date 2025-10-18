from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from embedding import SentenceTransformerEmbedder
from llm import cluster_llm_callable, semantic_llm_callable
from pipeline import MindMapPipeline


def configure_pipeline() -> MindMapPipeline:
    embedder = SentenceTransformerEmbedder()
    return MindMapPipeline(
        embedder=embedder,
        cluster_llm=cluster_llm_callable,
        semantic_llm=semantic_llm_callable,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a reasoning graph from Chain-of-Thought text.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a text file containing a single Chain-of-Thought trace.",
    )
    parser.add_argument(
        "--cluster-json",
        type=Path,
        nargs="*",
        default=(),
        help="Optional JSON files containing LLM-produced clustering outputs.",
    )
    parser.add_argument(
        "--semantics-json",
        type=Path,
        nargs="*",
        default=(),
        help="Optional JSON files containing adjacency samples.",
    )
    return parser


def load_payloads(paths: Sequence[Path]) -> list[str]:
    payloads: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        payloads.append(text)
    return payloads


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    text = args.input.read_text(encoding="utf-8")
    cluster_samples = load_payloads(args.cluster_json)
    semantic_samples = load_payloads(args.semantics_json)

    pipeline = configure_pipeline()
    result = pipeline.run(
        text,
        cluster_json_samples=cluster_samples,
        semantic_json_samples=semantic_samples,
    )

    print("# Reasoning Steps")
    for step in result.graph.steps:
        print(f"{step.key}: {step.title}")

    print("\n# Graph Metrics")
    metrics = result.metrics
    print(f"num_steps: {metrics.num_steps}")
    print(f"num_edges: {metrics.num_edges}")
    print(f"exploration_density: {metrics.exploration_density:.4f}")
    print(f"branching_ratio: {metrics.branching_ratio:.4f}")
    print(f"convergence_ratio: {metrics.convergence_ratio:.4f}")
    print(f"linearity: {metrics.linearity:.4f}")

    print("\n# Edge Confidence (signed)")
    for (src, dst), confidence in sorted(result.edge_confidence.items()):
        signed = confidence.signed_confidence()
        print(f"{src}->{dst}: {signed:.3f} (n={confidence.samples}, se={confidence.standard_error:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
