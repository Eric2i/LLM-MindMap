from __future__ import annotations

import json
import math
import threading
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ReasoningStep, ReasoningUnit

THINK_TOKEN_ID = 151668
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
MAX_NEW_TOKENS = 4096

_load_lock = threading.Lock()
_client: "QwenClient | None" = None


class QwenClient:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str, temperature: float) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generation = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=max(0.05, float(temperature)),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generation[0][inputs.input_ids.shape[1] :].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
        except ValueError:
            index = 0
        decoded = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return decoded


def _client_instance() -> QwenClient:
    global _client
    if _client is None:
        with _load_lock:
            if _client is None:
                _client = QwenClient()
    return _client


def cluster_llm_callable(
    units: Sequence[ReasoningUnit],
    temperature: float,
    sample_index: int,
) -> str:
    prompt = _build_cluster_prompt(units, temperature, sample_index)
    return _client_instance().generate(prompt, temperature)


def semantic_llm_callable(
    steps: Sequence[ReasoningStep],
    temperature: float,
    sample_index: int,
) -> str:
    prompt = _build_semantic_prompt(steps, temperature, sample_index)
    return _client_instance().generate(prompt, temperature)


def _build_cluster_prompt(
    units: Sequence[ReasoningUnit],
    temperature: float,
    sample_index: int,
) -> str:
    unit_lines = []
    for unit in units:
        unit_lines.append(f"{unit.index}: {unit.text.strip()}")
    unit_block = "\n".join(unit_lines)
    return (
        "You are clustering reasoning units into higher-level reasoning steps. "
        "Follow exactly the output format described below. "
        "Input reasoning units (ordered by index):\n"
        f"{unit_block}\n\n"
        "Return a JSON object where each key is s0, s1, ... in order, and each value has:\n"
        '  "title": concise title for the step,\n'
        '  "content": merged text of the grouped units,\n'
        '  "unit_indices": array of the original unit indices included in this step.\n'
        "Constraints:\n"
        "- Preserve the original order of units.\n"
        "- Each unit index must appear exactly once.\n"
        "- Keep steps contiguous (no skipping indices inside a step).\n"
        "- Do not include any explanation or markdown, only valid JSON.\n"
    )


def _build_semantic_prompt(
    steps: Sequence[ReasoningStep],
    temperature: float,
    sample_index: int,
) -> str:
    lines: List[str] = []
    for idx, step in enumerate(steps):
        lines.append(
            f"{step.key} (title: {step.title.strip()}): {step.content.strip()}"
        )
    step_block = "\n".join(lines)
    return (
        "You are analysing semantic relations between reasoning steps. "
        "For every ordered pair (si, sj) with i < j decide whether si supports, contradicts, "
        "or is independent of sj.\n"
        f"Reasoning steps:\n{step_block}\n\n"
        "Return a JSON object whose keys are string tuples such as \"(s0,s1)\" and whose values are "
        "one of: support, contradict, independent. Cover every i < j pair. "
        "Do not include explanations or markdown, only JSON.\n"
    )
