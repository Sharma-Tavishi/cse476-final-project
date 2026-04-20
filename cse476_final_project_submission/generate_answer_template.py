#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# Technique 1 — Chain of Thought
# ask the model to walk through the problem step by step before answering

def chain_of_thought(question: str, max_tokens: int = 700) -> str:
    system = "You are a careful problem solver. Think step by step."
    prompt = (
        f"{question}\n\n"
        "Work through this step by step. "
        "Write your final answer on the last line starting with 'Answer:'"
    )
    resp = call_llm(prompt, system=system, temperature=0.0, max_tokens=max_tokens)
    return extract_final_answer(resp)


# Technique 2 — Self-Consistency
# run the same question a few times at higher temperature, take the majority answer

def self_consistency(question: str, n: int = 5) -> str:
    from collections import Counter
    system = "You are a careful problem solver. Give only the final answer, nothing else."
    prompt = f"{question}\n\nFinal answer:"

    answers = []
    for _ in range(n):
        resp = call_llm(prompt, system=system, temperature=0.7, max_tokens=256)
        ans = resp.strip().split("\n")[0].strip()
        if ans:
            answers.append(ans)
        time.sleep(0.1)

    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]




# Technique 3 — Self-Refine
# give the model its own answer and ask it to spot and fix mistakes

def self_refine(question: str, initial_answer: str, max_tokens: int = 1024) -> str:
    prompt = (
        f"Question:\n{question}\n\n"
        f"Proposed answer:\n{initial_answer}\n\n"
        "Is there anything wrong or missing in this answer? "
        "If yes, fix it. Output only the corrected final answer, nothing else."
    )
    improved = call_llm(prompt, temperature=0.0, max_tokens=max_tokens)
    return improved.strip() if improved.strip() else initial_answer


# Technique 4 — Tree of Thought
# explore a few different reasoning paths, then pick the one that looks right

def tree_of_thought(question: str, branches: int = 3) -> str:
    branch_prompt = (
        f"{question}\n\n"
        f"Think of {branches} different ways to solve this. "
        "Number them 1 to 3. For each show the reasoning and the answer it leads to."
    )
    paths = call_llm(branch_prompt, temperature=0.5, max_tokens=1200)
    pick_prompt = (
        f"Problem: {question}\n\n"
        f"Here are {branches} different approaches:\n{paths}\n\n"
        "Which approach is most likely correct? "
        "Output only the final answer from the best approach, nothing else."
    )
    best = call_llm(pick_prompt, temperature=0.0, max_tokens=256)
    return best.strip()


# Technique 5 — ReACT (Reason + Act)
# the model thinks about what to do next, picks an action, then repeats
# good for planning problems where you need a sequence of steps

def react(question: str, max_steps: int = 5) -> str:
    system = (
        "You are a planner. At each step output:\n"
        "Thought: <what you're figuring out>\n"
        "Action: <the next action>\n"
        "When the plan is complete, output:\n"
        "Final Answer: <all actions in order, one per line>"
    )
    history = question + "\n\nLet's work through this step by step.\n"
    for _ in range(max_steps):
        resp = call_llm(history, system=system, temperature=0.0, max_tokens=512)
        history += resp + "\n"
        if "Final Answer:" in resp:
            return resp.split("Final Answer:")[-1].strip()
        time.sleep(0.1)
    return extract_final_answer(history)


# Technique 6 — Decomposition
# break the question into smaller pieces, answer each, then combine

def decompose_and_solve(question: str) -> str:
    decompose_prompt = (
        f"Question: {question}\n\n"
        "Break this into 2-3 simpler sub-questions. List them numbered."
    )
    sub_qs = call_llm(decompose_prompt, temperature=0.0, max_tokens=400)
    combine_prompt = (
        f"Main question: {question}\n\n"
        f"Sub-questions:\n{sub_qs}\n\n"
        "Answer each sub-question, then use those answers to give the final answer. "
        "End with 'Answer:' followed by just the final answer."
    )
    combined = call_llm(combine_prompt, temperature=0.0, max_tokens=700)
    return extract_final_answer(combined)


# Technique 7 — Tool-Augmented Reasoning
# for math: write python to solve it and actually run it
# for coding: verify the generated code compiles before returning

def tool_augmented_math(question: str) -> Optional[str]:
    code_prompt = (
        f"Solve this math problem by writing Python code:\n{question}\n\n"
        "Use only the standard library (math, itertools, fractions, etc.). "
        "Print the final answer on the last line. Output only the code."
    )
    code = call_llm(code_prompt, temperature=0.0, max_tokens=600)
    code = re.sub(r"```(?:python)?\n?", "", code).replace("```", "").strip()
    try:
        safe_env = {
            "math": __import__("math"),
            "itertools": __import__("itertools"),
            "fractions": __import__("fractions"),
            "print": print,
            "range": range, "len": len, "int": int, "float": float,
            "round": round, "abs": abs, "sum": sum, "min": min, "max": max,
            "sorted": sorted, "enumerate": enumerate, "zip": zip,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, safe_env)
        result = buf.getvalue().strip()
        if result:
            return result.splitlines()[-1].strip()
    except Exception:
        pass
    return None


def tool_augmented_code(question: str) -> str:
    gen_prompt = (
        f"{question}\n\n"
        "Write the Python function body only (not the def line). "
        "Output only the indented code."
    )
    code = call_llm(gen_prompt, temperature=0.0, max_tokens=900)
    try:
        compile(f"def _f():\n{code}", "<string>", "exec")
    except SyntaxError:
        fix_prompt = (
            f"This Python code has a syntax error:\n{code}\n\n"
            "Fix the syntax. Output only the corrected indented code."
        )
        code = call_llm(fix_prompt, temperature=0.0, max_tokens=900)
    return code.strip()


# Technique 8 — LLM-as-Judge
# when we have multiple candidate answers and need to pick the best one

def llm_judge(question: str, candidates: List[str]) -> str:
    if len(candidates) == 1:
        return candidates[0]
    options = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(candidates))
    judge_prompt = (
        f"Question: {question}\n\n"
        f"Candidate answers:\n{options}\n\n"
        "Which candidate answer is correct? Reply with only the number."
    )
    resp = call_llm(judge_prompt, temperature=0.0, max_tokens=10)
    m = re.search(r"\d+", resp)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


# --- domain solvers ----------------------------------------------------------

def solve_math(question: str) -> str:
    py_answer = tool_augmented_math(question)
    if py_answer:
        sc_answer = self_consistency(question, n=3)
        if sc_answer and normalize(py_answer) == normalize(sc_answer):
            return py_answer
        return llm_judge(question, [py_answer, sc_answer])
    sc_answer = self_consistency(question, n=5)
    if sc_answer:
        return sc_answer
    return tree_of_thought(question)


def solve_coding(question: str) -> str:
    initial = tool_augmented_code(question)
    return self_refine(question, initial, max_tokens=1024)


def solve_planning(question: str) -> str:
    return react(question, max_steps=5)


def solve_commonsense(question: str) -> str:
    is_complex = question.count("?") > 1 or len(question) > 400
    if is_complex:
        cot_ans    = chain_of_thought(question)
        decomp_ans = decompose_and_solve(question)
        return llm_judge(question, [cot_ans, decomp_ans])
    return chain_of_thought(question, max_tokens=400)


def solve_future(question: str) -> str:
    system = "You are a forecasting agent. Make your best prediction based on what you know."
    prompt = (
        f"{question}\n\n"
        "Think about this and make a prediction. "
        "Give your answer in the exact format the question asks for."
    )
    resp = call_llm(prompt, system=system, temperature=0.0, max_tokens=512)
    return extract_final_answer(resp)


# --- main agent --------------------------------------------------------------

def agent(question_text: str) -> str:
    domain = detect_domain(question_text)
    if domain == "math":
        return solve_math(question_text)
    elif domain == "coding":
        return solve_coding(question_text)
    elif domain == "planning":
        return solve_planning(question_text)
    elif domain == "future_prediction":
        return solve_future(question_text)
    else:
        return solve_commonsense(question_text)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions):
        if idx % 50 == 0:
            print(f"  progress: {idx}/{len(questions)}")
        try:
            answer = agent(question["input"])
        except Exception as e:
            print(f"  question {idx} failed ({e}), skipping")
            answer = ""
        answers.append({"output": str(answer)[:5000]})
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
