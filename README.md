# Multi-Domain Inference-Time Reasoning Agent

**CSE 476 Final Project** · Tavishi Sharma & Siya Garg 

An inference-time reasoning agent that solves heterogeneous natural language questions across five domains using a suite of eight distinct reasoning techniques. All LLM access is performed through an OpenAI-compatible API endpoint using `qwen3-30b-a3b-instruct-2507` hosted on ASU's SOL cluster.

---

## Overview

- **5 domains:** Mathematics, Coding, Planning, Common Sense, Future Prediction
- **8 inference-time techniques** composed into domain-specific solver pipelines
- **≤ 20 LLM calls per question** enforced via budget tracking
- **6,208 test questions** processed with incremental checkpointing and automatic resume

---

## Inference-Time Techniques

| # | Technique | Description |
|---|-----------|-------------|
| 1 | **Chain of Thought (CoT)** | Step-by-step reasoning before committing to a final `Answer:` |
| 2 | **Self-Consistency** | Samples `n` diverse reasoning paths (temp 0.7), selects majority answer |
| 3 | **Self-Refine** | Follow-up call critiques and refines the model's own proposed answer |
| 4 | **Tree of Thought (ToT)** | Generates 3 solution branches, then selects the best via a second call |
| 5 | **ReACT** | Iterative `Thought:/Action:` loop (up to 5 steps) for sequential planning |
| 6 | **Decomposition** | Breaks question into 2–3 sub-questions, answers each, then synthesizes |
| 7 | **Tool-Augmented Reasoning** | Executes sandboxed Python (math) or compile-checks generated code (coding) |
| 8 | **LLM-as-Judge** | A separate LLM call selects among conflicting candidate answers |

---

## Domain Solver Routing

| Domain | Primary Technique(s) | Fallback / Adjudication |
|--------|----------------------|-------------------------|
| Math | Tool-augmented (Python exec) + Self-consistency | LLM-as-Judge → Tree of Thought |
| Coding | Tool-augmented (compile check) | Self-refine |
| Planning | ReACT loop (up to 5 steps) | CoT extraction |
| Common Sense | CoT (short questions) | Decomposition + LLM-as-Judge |
| Future Prediction | CoT with forecasting system prompt | — |

### Math Solver Flow

```
Math question
    │
    ▼
Tool-augmented (Python exec)
    │
    ├─ Code ran OK? ──Yes──► Self-consistency (n=3)
    │                              │
    │                         Answers agree? ──Yes──► Return answer
    │                              │
    │                              No
    │                              ▼
    └─ No ──────────────────► Self-consistency (n=5)
                                   │
                              Answers agree? ──Yes──► Return answer
                                   │
                                   No
                                   ▼
                              LLM-as-Judge
                                   │
                                   ▼
                              Tree of Thought
```

---

## Domain Detection

A lightweight rule-based classifier assigns each question to a domain before any reasoning is applied, using keyword and pattern matching:

- **Math** — `$` (LaTeX), keywords: `find the`, `how many`, `compute`, `calculate`, `triangle`, etc.
- **Coding** — signals: `function`, `implement`, `write a program`, `returns the`
- **Planning** — phrases: `actions i can do`, logistics keywords (`truck`, `hoist`)
- **Future Prediction** — phrases: `predict future events`, `\boxed` markup
- **Common Sense** — default when no stronger signal is found

---

## Infrastructure & Robustness

- **HTTP retries with exponential backoff** — up to 5 retries on 429/500/502/503/504, with `sleep(2^attempt)` delays
- **Separate connect/read timeouts** — 10s connect, 120s read
- **Code execution timeout** — 8-second `SIGALRM` wall-clock limit on sandboxed Python exec; `finally` block clears alarm
- **Incremental checkpointing** — answers serialized to disk after every question; auto-resumes from last completed index on restart
- **LLM call budget tracking** — global counter incremented on every call, reported in live progress display
- **Dry-run mode** — `--dry-run` flag bypasses all API calls for control-flow testing
- **Local model mode** — `--local` flag redirects to a local LM Studio instance on `localhost:1234`

---

## Answer Post-Processing

All model outputs pass through `extract_final_answer` with three rules applied in priority order:

1. Split on the last `Answer:` marker
2. Capture a `\boxed{...}` LaTeX expression
3. Return the last non-empty line of the response

Outputs are capped at 5,000 characters per the grader format specification.

---

## Project Structure

```
generate_answer_template.py   # Main agent script
├── detect_domain()           # Rule-based domain classifier (lines 113–132)
├── call_llm()                # Shared LLM wrapper with retry logic (line 58+)
├── extract_final_answer()    # Answer extraction helper (lines 97–105)
├── chain_of_thought()        # CoT technique (lines 143–151)
├── self_consistency()        # Self-consistency technique (lines 157–172)
├── self_refine()             # Self-refine technique (lines 180–188)
├── tree_of_thought()         # ToT technique (lines 194–208)
├── react()                   # ReACT technique (lines 215–230)
├── decompose()               # Decomposition technique (lines 236–249)
├── tool_augmented_math()     # Python exec sandbox (lines 256–289)
├── tool_augmented_coding()   # Compile check + self-repair (lines 322–337)
├── llm_as_judge()            # Judge technique (lines 343–358)
├── agent()                   # Main dispatcher (lines 407–418)
└── build_answers()           # Checkpointed batch runner (lines 429–447)
```

---

## Usage

```bash
# Normal run
python generate_answer_template.py

# Test control flow without consuming API quota
python generate_answer_template.py --dry-run

# Use local LM Studio instance
python generate_answer_template.py --local
```
