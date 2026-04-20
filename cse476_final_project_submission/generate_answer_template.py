#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")


def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    return ""


def extract_final_answer(text: str) -> str:
    text = text.strip()
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip().split("\n")[0].strip()
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def detect_domain(text: str) -> str:
    t = text.lower()

    if "predict future events" in t or "请预测" in t or "\\boxed" in text:
        return "future_prediction"

    if "actions i can do" in t or ("logistics" in t and "truck" in t) or "hoist" in t:
        return "planning"

    coding_signals = ["function", "implement", "write a program", ">>> ",
                      "returns the", "retrieves the", "generates a"]
    if any(s in t for s in coding_signals):
        return "coding"

    math_signals = ["find the", "how many", "let $", "triangle", "probability",
                    "integer", "\\frac", "\\sqrt", "compute", "calculate"]
    if "$" in text or any(s in t for s in math_signals):
        return "math"

    return "common_sense"


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions, start=1):
        # Example: assume you have an agent loop that produces an answer string.
        # real_answer = agent_loop(question["input"])
        # answers.append({"output": real_answer})
        placeholder_answer = f"Placeholder answer for question {idx}"
        answers.append({"output": placeholder_answer})
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()
