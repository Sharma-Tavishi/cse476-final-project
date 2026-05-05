from __future__ import annotations

"""
CSE 476 Final Project - General-Purpose Reasoning Agent

This agent implements 8+ inference-time techniques:
1. Chain of Thought (CoT)
2. Self-Consistency  
3. Self-Refine
4. Tree of Thoughts (ToT)
5. ReACT (Reasoning + Acting)
6. Decomposition
7. Tool-Augmented Reasoning
8. LLM-as-Judge

Author: CSE 476 Team
Date: April 2026
"""

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
# CONFIGURATION
# =============================================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

MAX_CALLS_PER_QUESTION = 20
SELF_CONSISTENCY_SAMPLES = 2
MAX_TOKENS = 512

# Call tracking
_call_counter = 0


# =============================================================================
# CORE API FUNCTIONS
# =============================================================================

def reset_call_counter() -> None:
    global _call_counter
    _call_counter = 0


def get_call_count() -> int:
    return _call_counter


def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    silent: bool = False,
) -> str:
    """Make an LLM API call with call counting."""
    global _call_counter
    _call_counter += 1
    
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
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"] or ""
        return ""
    except Exception:
        return ""


# =============================================================================
# ANSWER PROCESSING
# =============================================================================

def extract_final_answer(text: str) -> str:
    """Extract final answer from model output."""
    if not text:
        return ""
    
    text = text.strip()
    
    if "Answer:" in text:
        parts = text.split("Answer:")
        if len(parts) > 1:
            return parts[-1].strip().split("\n")[0].strip()
    
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    
    if "Final answer:" in text.lower():
        parts = re.split(r"Final answer:", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip().split("\n")[0].strip()
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


def clean_answer(answer: str) -> str:
    """Clean up the final answer."""
    if not answer:
        return ""
    
    answer = answer.strip()
    prefixes = ["Answer:", "Final answer:", "The answer is", "Therefore,", "Thus,"]
    
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            if answer.startswith(":"):
                answer = answer[1:].strip()
            break
    
    if answer and answer[-1] == '.' and answer[:-1].replace('-', '').isdigit():
        answer = answer[:-1]
    
    return answer.strip()


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def detect_domain(text: str) -> str:
    """Detect question domain."""
    t = text.lower()
    
    if "$" in text or any(s in t for s in ["find the", "how many", "compute", "solve"]):
        return "math"
    if any(s in t for s in ["function", "implement", "def ", "python", "code"]):
        return "coding"
    if any(s in t for s in ["predict", "forecast", "will happen"]):
        return "future"
    if any(s in t for s in ["actions", "logistics", "plan", "schedule"]):
        return "planning"
    
    return "commonsense"


# =============================================================================
# TECHNIQUE 1: Chain of Thought (1 call)
# =============================================================================

def chain_of_thought(question: str) -> str:
    """Think step by step before answering."""
    system = "Think step by step. End with 'Answer:'"
    prompt = f"{question}\n\nWork step by step. End with 'Answer:'"
    resp = call_llm(prompt, system=system, temperature=0.0)
    return clean_answer(extract_final_answer(resp))


# =============================================================================
# TECHNIQUE 2: Self-Consistency (N calls)
# =============================================================================

def self_consistency(question: str, n: int = None) -> str:
    """Multiple samples, take majority answer."""
    if n is None:
        n = SELF_CONSISTENCY_SAMPLES
    
    system = "Give only the final answer."
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
    norm_answers = [(a, normalize(a)) for a in answers]
    norm_counts = Counter(n for _, n in norm_answers)
    
    if norm_counts:
        most_common = norm_counts.most_common(1)[0][0]
        for orig, norm in norm_answers:
            if norm == most_common:
                return clean_answer(orig)
    
    return clean_answer(answers[0])


# =============================================================================
# TECHNIQUE 3: Self-Refine (1 call)
# =============================================================================

def self_refine(question: str, initial_answer: str) -> str:
    """Critique and improve an answer."""
    if not initial_answer:
        return ""
    
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {initial_answer}\n\n"
        "Is this correct? If not, fix it. Output only the corrected answer."
    )
    improved = call_llm(prompt, temperature=0.0)
    return clean_answer(improved.strip() or initial_answer)


# =============================================================================
# TECHNIQUE 4: Tree of Thoughts (2 calls)
# =============================================================================

def tree_of_thought(question: str) -> str:
    """Explore multiple reasoning paths."""
    branches = 2
    branch_prompt = (
        f"{question}\n\n"
        f"Think of {branches} different ways to solve this. "
        f"Number them. Show reasoning and answer for each."
    )
    paths = call_llm(branch_prompt, temperature=0.5, max_tokens=800)
    
    pick_prompt = (
        f"Problem: {question}\n\n"
        f"Approaches:\n{paths}\n\n"
        "Which approach is correct? Output only the final answer."
    )
    best = call_llm(pick_prompt, temperature=0.0, max_tokens=256)
    return clean_answer(best.strip())


# =============================================================================
# TECHNIQUE 5: ReACT (up to 4 calls)
# =============================================================================

def react(question: str) -> str:
    """Reasoning + Acting loop for planning."""
    system = (
        "At each step output:\n"
        "Thought: <what you're figuring out>\n"
        "Action: <next action>\n"
        "When complete, output:\n"
        "Final Answer: <answer>"
    )
    history = question + "\n\nLet's work through this.\n"
    
    for _ in range(3):
        resp = call_llm(history, system=system, temperature=0.0, max_tokens=512, silent=True)
        history += resp + "\n"
        if "Final Answer:" in resp:
            return clean_answer(resp.split("Final Answer:")[-1].strip())
        time.sleep(0.05)
    
    return clean_answer(extract_final_answer(history))


# =============================================================================
# TECHNIQUE 6: Decomposition (2 calls)
# =============================================================================

def decompose(question: str) -> str:
    """Break problem into sub-questions."""
    decomp_prompt = f"Question: {question}\n\nBreak this into 2-3 simpler sub-questions."
    sub_qs = call_llm(decomp_prompt, temperature=0.0, max_tokens=400)
    
    combine_prompt = (
        f"Main question: {question}\n\n"
        f"Sub-questions:\n{sub_qs}\n\n"
        "Answer each, then give final answer starting with 'Answer:'"
    )
    combined = call_llm(combine_prompt, temperature=0.0, max_tokens=700)
    return clean_answer(extract_final_answer(combined))


# =============================================================================
# TECHNIQUE 7: Tool-Augmented (1-2 calls)
# =============================================================================

def tool_math(question: str) -> Optional[str]:
    """Solve math with Python code."""
    code_prompt = (
        f"Solve with Python code:\n{question}\n\n"
        "Print the answer. Output only the code."
    )
    code = call_llm(code_prompt, temperature=0.0, max_tokens=400, silent=True)
    code = re.sub(r"```(?:python)?\n?", "", code).replace("```", "").strip()
    
    if not code:
        return None
    
    try:
        safe_env = {
            "math": __import__("math"),
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "sum": sum,
            "min": min,
            "max": max,
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


def tool_code(question: str) -> str:
    """Generate and validate code."""
    gen_prompt = (
        f"{question}\n\n"
        "Write the function body only. Output only the indented code."
    )
    code = call_llm(gen_prompt, temperature=0.0, max_tokens=600, silent=True)
    code = code.strip()
    
    # Validate syntax
    try:
        compile(f"def _f():\n{code}", "<string>", "exec")
    except SyntaxError:
        fix_prompt = f"Fix syntax error:\n{code}\n\nOutput only corrected code."
        code = call_llm(fix_prompt, temperature=0.0, max_tokens=600, silent=True)
        code = code.strip()
    
    return code


# =============================================================================
# TECHNIQUE 8: LLM-as-Judge (1 call)
# =============================================================================

def llm_judge(question: str, candidates: List[str]) -> str:
    """Select best answer from candidates."""
    if len(candidates) <= 1:
        return clean_answer(candidates[0]) if candidates else ""
    
    options = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {question}\n\nCandidates:\n{options}\n\n"
        "Which is correct? Reply with only the number."
    )
    resp = call_llm(prompt, temperature=0.0, max_tokens=10, silent=True)
    
    m = re.search(r"\d+", resp)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates):
            return clean_answer(candidates[idx])
    
    return clean_answer(candidates[0])


# =============================================================================
# DOMAIN SOLVERS
# =============================================================================

def solve_math(question: str) -> str:
    """Solve math problems."""
    if len(question) < 100:
        return chain_of_thought(question)
    
    py_answer = tool_math(question)
    if py_answer:
        sc_answer = self_consistency(question, n=2)
        if sc_answer and normalize(py_answer) == normalize(sc_answer):
            return py_answer
        return llm_judge(question, [py_answer, sc_answer])
    
    return self_consistency(question, n=3)


def solve_coding(question: str) -> str:
    """Solve coding problems."""
    code = tool_code(question)
    if code:
        return self_refine(question, code)
    return chain_of_thought(question)


def solve_planning(question: str) -> str:
    """Solve planning problems."""
    return react(question)


def solve_future(question: str) -> str:
    """Solve prediction problems."""
    system = "Make a prediction. Give answer in exact format requested."
    prompt = f"{question}\n\nMake a prediction:"
    resp = call_llm(prompt, system=system, temperature=0.0)
    return clean_answer(extract_final_answer(resp))


def solve_commonsense(question: str) -> str:
    """Solve commonsense problems."""
    is_complex = question.count("?") > 1 or len(question) > 400
    if is_complex:
        return decompose(question)
    return chain_of_thought(question)


# =============================================================================
# MAIN AGENT
# =============================================================================

def agent(question_text: str) -> str:
    """Main agent entry point."""
    reset_call_counter()
    
    domain = detect_domain(question_text)
    
    if domain == "math":
        answer = solve_math(question_text)
    elif domain == "coding":
        answer = solve_coding(question_text)
    elif domain == "planning":
        answer = solve_planning(question_text)
    elif domain == "future":
        answer = solve_future(question_text)
    else:
        answer = solve_commonsense(question_text)
    
    return str(answer)[:5000] if answer else ""


# =============================================================================
# SUBMISSION FUNCTIONS
# =============================================================================

def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate answers for all questions."""
    answers = []
    total = len(questions)
    
    print(f"Processing {total} questions...")
    
    for idx, q in enumerate(questions):
        if idx % 100 == 0 and idx > 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        question_text = q.get("input") or q.get("prompt", "")
        try:
            answer = agent(question_text)
        except Exception:
            answer = ""
        
        answers.append({"output": answer[:5000]})
        time.sleep(0.05)
    
    return answers


def validate_results(questions: List[Dict], answers: List[Dict]) -> None:
    """Validate output format."""
    if len(questions) != len(answers):
        raise ValueError(f"Length mismatch: {len(questions)} vs {len(answers)}")
    for i, ans in enumerate(answers):
        if "output" not in ans:
            raise ValueError(f"Missing 'output' at index {i}")
        if not isinstance(ans["output"], str):
            raise ValueError(f"Non-string output at index {i}")
        if len(ans["output"]) >= 5000:
            raise ValueError(f"Answer {i} exceeds 5000 chars")


def main():
    """Main entry point."""
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print("\nWindows PowerShell:")
        print('  $env:OPENAI_API_KEY = "your-key-here"')
        print("\nMac/Linux:")
        print('  export OPENAI_API_KEY="your-key-here"')
        sys.exit(1)
    
    input_path = Path("cse_476_final_project_test_data.json")
    output_path = Path("cse_476_final_project_answers.json")
    
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)
    
    print("=" * 60)
    print("CSE 476 Final Project - Reasoning Agent")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Max calls/question: {MAX_CALLS_PER_QUESTION}")
    print("=" * 60)
    
    questions = load_questions(input_path)
    print(f"Loaded {len(questions)} questions")
    
    print("\nGenerating answers...")
    answers = build_answers(questions)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        validate_results(questions, json.load(f))
    
    print(f"\n✅ Success! Saved {len(answers)} answers to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()