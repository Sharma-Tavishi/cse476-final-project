#!/usr/bin/env python3
"""
General-Purpose Reasoning Agent for CSE 476 Final Project

Implements 8+ inference-time techniques:
1. Chain of Thought (CoT)
2. Self-Consistency
3. Self-Refine
4. Tree of Thoughts (ToT)
5. ReACT (Reasoning + Acting)
6. Decomposition
7. Tool-Augmented Reasoning (Math & Code)
8. LLM-as-Judge
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# =============================================================================
# Configuration
# =============================================================================

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Call tracking
_call_counter = 0
MAX_CALLS_PER_QUESTION = 20
_total_calls_all_questions = 0
_total_questions_processed = 0
_call_counts_per_question = []


# =============================================================================
# Core API Functions
# =============================================================================

def reset_call_counter() -> None:
    """Reset the call counter for a new question."""
    global _call_counter
    _call_counter = 0


def get_call_count() -> int:
    """Get current number of LLM calls made."""
    return _call_counter


def add_to_statistics() -> None:
    """Add current question's call count to statistics."""
    global _total_calls_all_questions, _total_questions_processed, _call_counts_per_question
    _total_calls_all_questions += _call_counter
    _total_questions_processed += 1
    _call_counts_per_question.append(_call_counter)


def print_statistics(questions_processed: int, prefix: str = "") -> None:
    """Print current call statistics."""
    if questions_processed == 0:
        return
    
    avg_calls = _total_calls_all_questions / questions_processed
    max_calls = max(_call_counts_per_question[-questions_processed:]) if _call_counts_per_question else 0
    min_calls = min(_call_counts_per_question[-questions_processed:]) if _call_counts_per_question else 0
    over_limit = sum(1 for c in _call_counts_per_question[-questions_processed:] if c > MAX_CALLS_PER_QUESTION)
    
    print(f"{prefix}Statistics (last {questions_processed} questions):")
    print(f"  ├─ Total LLM calls: {_total_calls_all_questions}")
    print(f"  ├─ Average calls/question: {avg_calls:.1f}")
    print(f"  ├─ Min calls: {min_calls}")
    print(f"  ├─ Max calls: {max_calls}")
    print(f"  └─ Questions exceeding {MAX_CALLS_PER_QUESTION} calls: {over_limit}")


def check_call_limit() -> None:
    """Check if we've exceeded the call limit and warn."""
    if _call_counter > MAX_CALLS_PER_QUESTION:
        print(f"WARNING: Exceeded {MAX_CALLS_PER_QUESTION} calls ({_call_counter} used)")


def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    silent: bool = False,
) -> str:
    """Make an LLM API call with call counting."""
    global _call_counter
    _call_counter += 1
    
    if not silent and _call_counter > MAX_CALLS_PER_QUESTION:
        print(f"  Warning: Call {_call_counter} exceeds limit of {MAX_CALLS_PER_QUESTION}")
    
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        else:
            if not silent:
                print(f"  API error: {resp.status_code}")
            return ""
    except Exception as e:
        if not silent:
            print(f"  API exception: {e}")
        return ""


# =============================================================================
# Answer Processing Helpers
# =============================================================================

def extract_final_answer(text: str) -> str:
    """Extract final answer from model output."""
    if not text:
        return ""
    
    text = text.strip()
    
    # Look for "Answer:" pattern
    if "Answer:" in text:
        parts = text.split("Answer:")
        if len(parts) > 1:
            answer = parts[-1].strip().split("\n")[0].strip()
            if answer:
                return answer
    
    # Look for LaTeX boxed format
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    
    # Look for "Final answer:" pattern
    if "Final answer:" in text.lower():
        parts = re.split(r"Final answer:", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            answer = parts[-1].strip().split("\n")[0].strip()
            if answer:
                return answer
    
    # Take last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


def clean_final_answer(answer: str) -> str:
    """Clean up the final answer by removing common prefixes."""
    if not answer:
        return ""
    
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes = [
        "Answer:", "Final answer:", "The answer is", "Answer=",
        "Therefore,", "Thus,", "So,", "Hence,", "Result:"
    ]
    
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            if answer.startswith(":"):
                answer = answer[1:].strip()
            break
    
    # Remove trailing period from numbers
    if answer and answer[-1] == '.' and answer[:-1].replace('-', '').replace('.', '').isdigit():
        answer = answer[:-1]
    
    return answer.strip()


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def detect_domain(text: str) -> str:
    """Detect the domain of a question."""
    t = text.lower()

    if "predict future events" in t or "请预测" in t or "will happen" in t:
        return "future_prediction"

    if "actions i can do" in t or ("logistics" in t and "truck" in t) or "hoist" in t:
        return "planning"

    coding_signals = ["function", "implement", "write a program", ">>> ",
                      "returns the", "retrieves the", "generates a", "def ",
                      "python", "code"]
    if any(s in t for s in coding_signals):
        return "coding"

    math_signals = ["find the", "how many", "let $", "triangle", "probability",
                    "integer", "\\frac", "\\sqrt", "compute", "calculate",
                    "solve", "equation"]
    if "$" in text or any(s in t for s in math_signals):
        return "math"

    return "common_sense"


# =============================================================================
# Technique 1: Chain of Thought (CoT)
# =============================================================================

def chain_of_thought(question: str, max_tokens: int = 700) -> str:
    """Chain of Thought reasoning - 1 call."""
    system = "You are a careful problem solver. Think step by step."
    prompt = (
        f"{question}\n\n"
        "Work through this step by step. "
        "Write your final answer on the last line starting with 'Answer:'"
    )
    resp = call_llm(prompt, system=system, temperature=0.0, max_tokens=max_tokens)
    answer = extract_final_answer(resp)
    return clean_final_answer(answer)


# =============================================================================
# Technique 2: Self-Consistency
# =============================================================================

def self_consistency(question: str, n: int = 3) -> str:
    """Self-consistency sampling - n calls."""
    system = "You are a careful problem solver. Give only the final answer, nothing else."
    prompt = f"{question}\n\nFinal answer:"

    answers = []
    for _ in range(n):
        resp = call_llm(prompt, system=system, temperature=0.7, max_tokens=256, silent=True)
        ans = resp.strip().split("\n")[0].strip()
        if ans:
            answers.append(ans)
        time.sleep(0.05)

    if not answers:
        return ""

    # Find most common answer
    normalized_answers = [(a, normalize(a)) for a in answers]
    normalized_counts = Counter(n for _, n in normalized_answers)
    
    if normalized_counts:
        most_common_norm = normalized_counts.most_common(1)[0][0]
        for orig, norm in normalized_answers:
            if norm == most_common_norm:
                return clean_final_answer(orig)
    
    return clean_final_answer(answers[0])


# =============================================================================
# Technique 3: Self-Refine
# =============================================================================

def self_refine(question: str, initial_answer: str, max_tokens: int = 1024) -> str:
    """Self-refine: critique and improve answer - 1 call."""
    if not initial_answer:
        return ""
    
    prompt = (
        f"Question:\n{question}\n\n"
        f"Proposed answer:\n{initial_answer}\n\n"
        "Is there anything wrong or missing in this answer? "
        "If yes, fix it. Output only the corrected final answer, nothing else."
    )
    improved = call_llm(prompt, temperature=0.0, max_tokens=max_tokens)
    improved = improved.strip()
    
    if improved:
        return clean_final_answer(improved)
    return clean_final_answer(initial_answer)


# =============================================================================
# Technique 4: Tree of Thoughts (ToT)
# =============================================================================

def tree_of_thought(question: str, branches: int = 3) -> str:
    """Tree of Thoughts - 2 calls."""
    branch_prompt = (
        f"{question}\n\n"
        f"Think of {branches} different ways to solve this. "
        f"Number them 1 to {branches}. For each, show the reasoning and the answer."
    )
    paths = call_llm(branch_prompt, temperature=0.5, max_tokens=1200)

    pick_prompt = (
        f"Problem: {question}\n\n"
        f"Different approaches:\n{paths}\n\n"
        "Which approach is most likely correct? "
        "Output only the final answer from the best approach, nothing else."
    )
    best = call_llm(pick_prompt, temperature=0.0, max_tokens=256)
    return clean_final_answer(best.strip())


# =============================================================================
# Technique 5: ReACT (Reasoning + Acting)
# =============================================================================

def react(question: str, max_steps: int = 5) -> str:
    """ReACT loop for planning - up to max_steps calls."""
    system = (
        "You are a planner. At each step output:\n"
        "Thought: <what you're figuring out>\n"
        "Action: <the next action>\n"
        "When the plan is complete, output:\n"
        "Final Answer: <all actions in order, one per line>"
    )
    history = question + "\n\nLet's work through this step by step.\n"

    for _ in range(max_steps):
        resp = call_llm(history, system=system, temperature=0.0, max_tokens=512, silent=True)
        history += resp + "\n"
        if "Final Answer:" in resp:
            return clean_final_answer(resp.split("Final Answer:")[-1].strip())
        time.sleep(0.05)

    return clean_final_answer(extract_final_answer(history))


# =============================================================================
# Technique 6: Decomposition
# =============================================================================

def decompose_and_solve(question: str) -> str:
    """Decompose problem into sub-questions - 2 calls."""
    decompose_prompt = (
        f"Question: {question}\n\n"
        "Break this into 2-3 simpler sub-questions. List them numbered."
    )
    sub_qs = call_llm(decompose_prompt, temperature=0.0, max_tokens=400)

    combine_prompt = (
        f"Main question: {question}\n\n"
        f"Sub-questions:\n{sub_qs}\n\n"
        "Answer each sub-question, then give the final answer. "
        "End with 'Answer:' followed by just the final answer."
    )
    combined = call_llm(combine_prompt, temperature=0.0, max_tokens=700)
    return clean_final_answer(extract_final_answer(combined))


# =============================================================================
# Technique 7: Tool-Augmented Reasoning
# =============================================================================

def tool_augmented_math(question: str) -> Optional[str]:
    """Solve math with Python code - 1 call."""
    code_prompt = (
        f"Solve this math problem by writing Python code:\n{question}\n\n"
        "Use only the standard library. Print the final answer. Output only the code."
    )
    code = call_llm(code_prompt, temperature=0.0, max_tokens=600, silent=True)
    code = re.sub(r"```(?:python)?\n?", "", code).replace("```", "").strip()

    if not code:
        return None

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
    """Generate code for coding problems - 1-2 calls."""
    gen_prompt = (
        f"{question}\n\n"
        "Write the Python function body only (not the def line). "
        "Output only the indented code."
    )
    code = call_llm(gen_prompt, temperature=0.0, max_tokens=900, silent=True)
    code = code.strip()

    # Validate syntax
    try:
        compile(f"def _f():\n{code}", "<string>", "exec")
    except SyntaxError:
        fix_prompt = (
            f"This code has a syntax error:\n{code}\n\n"
            "Fix the syntax. Output only the corrected indented code."
        )
        code = call_llm(fix_prompt, temperature=0.0, max_tokens=900, silent=True)
        code = code.strip()

    return code


# =============================================================================
# Technique 8: LLM-as-Judge
# =============================================================================

def llm_judge(question: str, candidates: List[str]) -> str:
    """Use LLM to select best answer - 1 call."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return clean_final_answer(candidates[0])

    options = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    judge_prompt = (
        f"Question: {question}\n\n"
        f"Candidate answers:\n{options}\n\n"
        "Which candidate answer is correct? Reply with only the number."
    )
    resp = call_llm(judge_prompt, temperature=0.0, max_tokens=10, silent=True)
    
    m = re.search(r"\d+", resp)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates):
            return clean_final_answer(candidates[idx])
    
    return clean_final_answer(candidates[0])


# =============================================================================
# Domain-Specific Solvers
# =============================================================================

def solve_math(question: str) -> str:
    """Solve math problems - 2-5 calls."""
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
    """Solve coding problems - 2-3 calls."""
    initial = tool_augmented_code(question)
    if initial:
        refined = self_refine(question, initial, max_tokens=1024)
        return refined
    return chain_of_thought(question)


def solve_planning(question: str) -> str:
    """Solve planning problems - 3-6 calls."""
    return react(question, max_steps=5)


def solve_commonsense(question: str) -> str:
    """Solve common sense problems - 2-3 calls."""
    is_complex = question.count("?") > 1 or len(question) > 400
    if is_complex:
        cot_ans = chain_of_thought(question)
        decomp_ans = decompose_and_solve(question)
        return llm_judge(question, [cot_ans, decomp_ans])
    
    return chain_of_thought(question, max_tokens=400)


def solve_future(question: str) -> str:
    """Solve future prediction problems - 1 call."""
    system = "You are a forecasting agent. Make your best prediction."
    prompt = (
        f"{question}\n\n"
        "Think about this and make a prediction. "
        "Give your answer in the exact format the question asks for."
    )
    resp = call_llm(prompt, system=system, temperature=0.0, max_tokens=512)
    return clean_final_answer(extract_final_answer(resp))


# =============================================================================
# Main Agent Loop
# =============================================================================

def agent(question_text: str, verbose: bool = False) -> str:
    """Main agent entry point."""
    reset_call_counter()
    
    if verbose:
        print(f"  Processing question...")
    
    domain = detect_domain(question_text)
    
    if verbose:
        print(f"  Domain: {domain}")
    
    # Route to appropriate solver
    if domain == "math":
        answer = solve_math(question_text)
    elif domain == "coding":
        answer = solve_coding(question_text)
    elif domain == "planning":
        answer = solve_planning(question_text)
    elif domain == "future_prediction":
        answer = solve_future(question_text)
    else:
        answer = solve_commonsense(question_text)
    
    if answer is None:
        answer = ""
    
    answer = str(answer)[:5000]
    
    if verbose:
        print(f"  Calls used: {get_call_count()}")
        if get_call_count() > MAX_CALLS_PER_QUESTION:
            print(f"  WARNING: Exceeded call limit!")
    
    add_to_statistics()
    
    return answer


# =============================================================================
# Required Functions
# =============================================================================

def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build answers for all questions."""
    global _total_calls_all_questions, _total_questions_processed, _call_counts_per_question
    
    # Reset statistics
    _total_calls_all_questions = 0
    _total_questions_processed = 0
    _call_counts_per_question = []
    
    answers = []
    total = len(questions)
    
    print(f"Processing {total} questions...")
    print("-" * 60)
    
    for idx, question in enumerate(questions):
        # Progress update
        if idx % 100 == 0 and idx > 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        try:
            input_text = question.get("input") or question.get("prompt", "")
            if not input_text:
                print(f"  Warning: Question {idx} has no input/prompt field")
                answer = ""
            else:
                answer = agent(input_text, verbose=False)
        except Exception as e:
            print(f"  Error on question {idx}: {e}")
            answer = ""
        
        # Truncate if needed
        if len(answer) > 5000:
            answer = answer[:4997] + "..."
        
        answers.append({"output": answer})
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total questions processed: {total}")
    if total > 0:
        avg_calls = _total_calls_all_questions / total
        print(f"Total LLM calls: {_total_calls_all_questions}")
        print(f"Average calls/question: {avg_calls:.1f}")
        print(f"Max calls: {max(_call_counts_per_question) if _call_counts_per_question else 0}")
        print(f"Min calls: {min(_call_counts_per_question) if _call_counts_per_question else 0}")
        
        over_limit = sum(1 for c in _call_counts_per_question if c > MAX_CALLS_PER_QUESTION)
        if over_limit > 0:
            print(f"WARNING: {over_limit} questions exceeded {MAX_CALLS_PER_QUESTION} calls!")
    print(f"{'='*60}\n")
    
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    """Validate answers format."""
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
                f"({len(answer['output'])} chars)."
            )


def main() -> None:
    """Main entry point."""
    # Check API key
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    # Check input file
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("CSE 476 Final Project - Reasoning Agent")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Max calls per question: {MAX_CALLS_PER_QUESTION}")
    print("=" * 60)
    
    # Load questions
    print("\nLoading questions...")
    questions = load_questions(INPUT_PATH)
    print(f"Loaded {len(questions)} questions")
    
    # Generate answers
    print("\nGenerating answers...")
    answers = build_answers(questions)
    
    # Write output
    print(f"\nWriting answers to {OUTPUT_PATH}...")
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    
    # Validate
    print("Validating output format...")
    with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    
    print(f"\n Success! Wrote {len(answers)} answers to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()