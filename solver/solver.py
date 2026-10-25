#!/usr/bin/env python3
"""
ARC-AGI-1 Solver (Educational & Comprehensive)
==============================================
Evaluates Google Gemini 3.5 Flash Lite (reasoning: high) on 20 ARC-AGI-1 tasks
using two distinct paradigms:
  1. End-to-End (Fim a Fim): Direct grid inference via LLM reasoning.
  2. Code Generation (Geração de Código com Laço de Correção): Generating a Python
     `transform(grid)` function validated on training examples in a secure sandbox
     with iterative diagnostic feedback and correction loop (up to max_attempts).

Author: Mike Carvalho Borges
Institution: Especialização em Inteligência Artificial Avançada, UFRGS
Email: mike.borges@ufrgs.br
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai

# Ensure unbuffered, real-time terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TaskResult:
    """Stores detailed evaluation results for a single task and approach."""
    task_id: str
    approach: str  # "end_to_end" or "code"
    input_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    total_tokens: int = 0
    api_time_seconds: float = 0.0
    exec_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    attempts: int = 1
    train_passed: int = 0
    train_total: int = 0
    test_passed: int = 0
    test_total: int = 0
    solved: bool = False
    solved_with_loop: bool = False
    perfect_train_failed_test: bool = False  # Acerta o treino e erra o teste
    grade: float = 0.0  # 1.0 if all test cases solved, else 0.0
    error_message: Optional[str] = None
    generated_code: Optional[str] = None
    predictions: List[List[List[int]]] = field(default_factory=list)


@dataclass
class ExperimentSummary:
    """Stores aggregated metrics across all evaluated tasks."""
    label: str
    model: str
    thinking_level: str
    approach: str
    timestamp: str
    total_tasks: int = 0
    solved_tasks: int = 0
    accuracy_percentage: float = 0.0
    total_attempts: int = 0
    avg_attempts_per_task: float = 1.0
    solved_with_loop_count: int = 0
    perfect_train_failed_test_count: int = 0
    perfect_train_failed_test_tasks: List[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_answer_tokens: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_task: float = 0.0
    tasks: List[TaskResult] = field(default_factory=list)


# ==============================================================================
# Environment & Client Setup
# ==============================================================================

def setup_gemini_client() -> genai.Client:
    """
    Locates .env in current or parent directories and initializes Gemini Client.
    """
    cur = pathlib.Path.cwd().resolve()
    loaded = False
    while cur != cur.parent:
        candidate = cur / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            loaded = True
            break
        cur = cur.parent

    if not loaded:
        load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API Key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    return genai.Client(api_key=api_key)


# ==============================================================================
# ARC Task Utilities
# ==============================================================================

def load_tasks_list(tasks_file_path: pathlib.Path) -> List[Tuple[str, pathlib.Path]]:
    """
    Reads tarefas.txt and returns list of (relative_name, absolute_path).
    """
    if not tasks_file_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file_path}")

    base_dir = tasks_file_path.parent
    tasks = []
    with open(tasks_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_path = base_dir / line
            if not task_path.exists():
                # Try relative to main workspace
                alt_path = base_dir.parent / line
                if alt_path.exists():
                    task_path = alt_path
            tasks.append((line, task_path))
    return tasks


def load_task_json(task_path: pathlib.Path) -> Dict[str, Any]:
    """Loads and validates a single ARC task JSON file."""
    with open(task_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "train" not in data or "test" not in data:
        raise ValueError(f"Invalid ARC task format in {task_path}")
    return data


def format_grid(grid: List[List[int]]) -> str:
    """Formats a 2D integer grid nicely for prompts."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    header = f"[{rows}x{cols} grid]"
    rows_str = ",\n ".join(json.dumps(row) for row in grid)
    return f"{header}\n[\n {rows_str}\n]"


def compare_grids(pred: Any, expected: Any) -> bool:
    """Strict element-by-element equality check between two 2D grids."""
    if not isinstance(pred, list) or not isinstance(expected, list):
        return False
    if len(pred) != len(expected):
        return False
    for r1, r2 in zip(pred, expected):
        if not isinstance(r1, list) or not isinstance(r2, list):
            return False
        if len(r1) != len(r2):
            return False
        for c1, c2 in zip(r1, r2):
            try:
                if int(c1) != int(c2):
                    return False
            except (ValueError, TypeError):
                return False
    return True


# ==============================================================================
# Prompt Engineering
# ==============================================================================

ARC_COLOR_MAP_DESCRIPTION = """
Colors are represented as integers 0 to 9:
  0: Black / Background
  1: Blue
  2: Red
  3: Green
  4: Yellow
  5: Gray
  6: Magenta
  7: Orange
  8: Teal / Azure
  9: Maroon / Brown
"""

def build_end_to_end_prompt(task: Dict[str, Any]) -> str:
    """Constructs prompt for End-to-End grid prediction."""
    parts = [
        "You are an expert AI solving the ARC-AGI (Abstraction and Reasoning Corpus) challenge.",
        "Your goal is to infer the underlying transformation rule from the training examples and predict the exact output grid for the test input.",
        ARC_COLOR_MAP_DESCRIPTION,
        "### Training Examples:",
    ]

    for i, pair in enumerate(task["train"], 1):
        parts.append(f"\n--- Example {i} ---")
        parts.append("Input:")
        parts.append(format_grid(pair["input"]))
        parts.append("Output:")
        parts.append(format_grid(pair["output"]))

    parts.append("\n### Test Problem:")
    for i, pair in enumerate(task["test"], 1):
        parts.append(f"Test Input {i}:")
        parts.append(format_grid(pair["input"]))

    parts.append("\n### Instructions:")
    parts.append(
        "1. Analyze the core geometric, topological, color, and object-level transformations.\n"
        "2. State your reasoning concisely.\n"
        "3. Provide the final predicted output grid strictly formatted inside a JSON code block as:\n"
        "```json\n"
        "{\n"
        '  "output": [[...], [...]]\n'
        "}\n"
        "```\n"
        "If there are multiple test inputs, format as a list of output grids: `\"output\": [[[...]], [[...]]]`."
    )
    return "\n".join(parts)


def build_code_gen_prompt(task: Dict[str, Any]) -> str:
    """Constructs initial prompt for Code Generation."""
    parts = [
        "You are an expert Python programmer and abstract reasoning specialist solving the ARC-AGI challenge.",
        "Your task is to write a standalone, correct Python function `transform(grid: list[list[int]]) -> list[list[int]]` that implements the transformation rule seen in the training examples.",
        ARC_COLOR_MAP_DESCRIPTION,
        "### Training Examples:",
    ]

    for i, pair in enumerate(task["train"], 1):
        parts.append(f"\n--- Example {i} ---")
        parts.append("Input:")
        parts.append(format_grid(pair["input"]))
        parts.append("Output:")
        parts.append(format_grid(pair["output"]))

    parts.append("\n### Test Problem:")
    for i, pair in enumerate(task["test"], 1):
        parts.append(f"Test Input {i} (Dimensions: {len(pair['input'])}x{len(pair['input'][0]) if pair['input'] else 0}):")
        parts.append(format_grid(pair["input"]))

    parts.append("\n### Implementation Requirements:")
    parts.append(
        "1. Define exactly `def transform(grid: list[list[int]]) -> list[list[int]]:`.\n"
        "2. The function must accept a 2D grid (list of lists of integers 0-9) and return a new 2D grid.\n"
        "3. You may use Python standard libraries (`copy`, `collections`, `itertools`, `math`, `heapq`) and `numpy`.\n"
        "4. Keep the function self-contained, pure, and robust.\n"
        "5. Output your code inside a single ```python ... ``` markdown code block."
    )
    return "\n".join(parts)


def build_correction_prompt(
    task: Dict[str, Any],
    previous_code: str,
    failed_details: List[Dict[str, Any]],
    passed_indices: List[int],
) -> str:
    """
    Constructs diagnostic feedback prompt for the correction loop (laço de correção).
    Shows which training examples failed and why, asking for a corrected function.
    """
    parts = [
        "You are an expert Python programmer and abstract reasoning specialist solving the ARC-AGI challenge.",
        "Your previous implementation of `def transform(grid)` FAILED on training example verification.",
        ARC_COLOR_MAP_DESCRIPTION,
        "\n### Previous Code Attempt:",
        f"```python\n{previous_code}\n```",
        "\n### Training Failure Diagnostics (Discrepancies):",
    ]

    for fail in failed_details:
        idx = fail["example_index"]
        parts.append(f"\n--- Train Example {idx} FAILED ---")
        parts.append("Input:")
        parts.append(format_grid(fail["input"]))
        parts.append("Expected Output:")
        parts.append(format_grid(fail["expected"]))
        if fail.get("actual") is not None:
            parts.append("Actual Output Produced by transform():")
            parts.append(format_grid(fail["actual"]))
        else:
            parts.append(f"Runtime Execution Error / Timeout: {fail.get('error')}")

    if passed_indices:
        parts.append(f"\nImportant Note: The previous code PASSED training examples: {passed_indices}. Do NOT break functionality for these cases.")

    parts.append("\n### Instructions for Correction:")
    parts.append(
        "1. Analyze why your previous rule or algorithm failed on the example(s) above.\n"
        "2. Identify the flaw in your logic (e.g. bounding box calculation, color replacement, object filtering, orientation, symmetry).\n"
        "3. Write a corrected, complete `transform(grid: list[list[int]]) -> list[list[int]]:` function.\n"
        "4. Output your code inside a single ```python ... ``` markdown code block."
    )
    return "\n".join(parts)


# ==============================================================================
# Parsers & Execution Sandbox
# ==============================================================================

def parse_end_to_end_predictions(response_text: str, num_test_cases: int) -> List[List[List[int]]]:
    """
    Extracts predicted grid(s) from model's response text.
    Handles multiple JSON schema variations robustly.
    """
    if not response_text:
        return []

    json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    candidates = json_blocks + [response_text]

    for cand in candidates:
        cand = cand.strip()
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and "output" in parsed:
                out = parsed["output"]
                if isinstance(out, list) and out:
                    if isinstance(out[0], list) and out[0] and isinstance(out[0][0], int):
                        return [out]
                    elif isinstance(out[0], list) and out[0] and isinstance(out[0][0], list):
                        return out
            elif isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], list) and parsed[0] and isinstance(parsed[0][0], int):
                    return [parsed]
        except Exception:
            continue

    array_matches = re.findall(r"\[\s*\[\s*\d[\d\s,]*\](?:\s*,\s*\[\s*\d[\d\s,]*\])*\s*\]", response_text)
    if array_matches:
        try:
            out = json.loads(array_matches[-1])
            if isinstance(out, list) and isinstance(out[0], list):
                return [out]
        except Exception:
            pass

    return []


def parse_code_block(response_text: str) -> Optional[str]:
    """Extracts python code from response."""
    if not response_text:
        return None
    matches = re.findall(r"```python\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if matches:
        for block in reversed(matches):
            if "def transform" in block:
                return block.strip()
        return matches[-1].strip()

    if "def transform" in response_text:
        return response_text.strip()
    return None


def execute_transform_in_sandbox(
    code: str,
    input_grid: List[List[int]],
    timeout_seconds: float = 5.0,
) -> Tuple[bool, Optional[List[List[int]]], float, str]:
    """
    Executes transform(input_grid) in an isolated subprocess with a strict time limit.
    Returns: (success, predicted_grid, execution_time_seconds, error_or_output_message)
    """
    runner_code = f"""import sys
import json
import copy
import math
from collections import deque, Counter, defaultdict
import itertools

try:
    import numpy as np
except ImportError:
    np = None

{code}

if __name__ == '__main__':
    try:
        raw_input = json.loads(sys.argv[1])
        output = transform(raw_input)
        if hasattr(output, 'tolist'):
            output = output.tolist()
        clean_out = [[int(cell) for cell in row] for row in output]
        print(json.dumps(clean_out))
    except Exception as e:
        sys.stderr.write(f"RuntimeError: {{e}}")
        sys.exit(1)
"""

    t_start = time.perf_counter()
    try:
        res = subprocess.run(
            [sys.executable, "-c", runner_code, json.dumps(input_grid)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - t_start

        if res.returncode == 0:
            try:
                output_grid = json.loads(res.stdout.strip())
                return True, output_grid, elapsed, "OK"
            except json.JSONDecodeError as jde:
                return False, None, elapsed, f"Invalid JSON output: {jde}"
        else:
            err = res.stderr.strip() or f"Process exited with code {res.returncode}"
            return False, None, elapsed, err

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t_start
        return False, None, elapsed, f"Execution timed out (> {timeout_seconds}s)"
    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        return False, None, elapsed, f"Execution failure: {exc}"


# ==============================================================================
# Single Task Solvers
# ==============================================================================

def solve_task_end_to_end(
    client: genai.Client,
    model: str,
    thinking_level: str,
    task_id: str,
    task: Dict[str, Any],
) -> TaskResult:
    """Evaluates a task using direct End-to-End inference."""
    result = TaskResult(task_id=task_id, approach="end_to_end")
    prompt = build_end_to_end_prompt(task)

    t0 = time.perf_counter()
    try:
        resp = client.interactions.create(
            model=model,
            input=prompt,
            generation_config={"thinking_level": thinking_level},
        )
        api_elapsed = time.perf_counter() - t0
        result.api_time_seconds = api_elapsed
        result.total_time_seconds = api_elapsed

        if resp.usage:
            result.input_tokens = resp.usage.total_input_tokens or 0
            result.reasoning_tokens = resp.usage.total_thought_tokens or 0
            result.answer_tokens = resp.usage.total_output_tokens or 0
            result.total_tokens = resp.usage.total_tokens or 0

        predictions = parse_end_to_end_predictions(resp.output_text or "", len(task["test"]))
        result.predictions = predictions

        num_test = len(task["test"])
        result.test_total = num_test
        passed = 0

        for i, test_pair in enumerate(task["test"]):
            expected = test_pair.get("output")
            if expected is None:
                continue
            if i < len(predictions) and compare_grids(predictions[i], expected):
                passed += 1

        result.test_passed = passed
        result.solved = (passed == num_test and num_test > 0)
        result.grade = 1.0 if result.solved else 0.0

    except Exception as exc:
        result.total_time_seconds = time.perf_counter() - t0
        result.error_message = str(exc)

    return result


def solve_task_code_gen(
    client: genai.Client,
    model: str,
    thinking_level: str,
    task_id: str,
    task: Dict[str, Any],
    timeout_seconds: float = 5.0,
    max_attempts: int = 3,
    delay_between_calls: float = 1.0,
) -> TaskResult:
    """
    Evaluates a task using Python Code Generation with a multi-turn Correction Loop (laço de correção).
    Validates on training pairs; if any fail, feeds diagnostic discrepancy back to model up to max_attempts.
    """
    result = TaskResult(task_id=task_id, approach="code")
    train_total = len(task["train"])
    test_total = len(task["test"])
    result.train_total = train_total
    result.test_total = test_total

    total_api_time = 0.0
    total_exec_time = 0.0

    current_code: Optional[str] = None
    failed_details: List[Dict[str, Any]] = []
    passed_indices: List[int] = []

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt

        if attempt == 1:
            prompt = build_code_gen_prompt(task)
        else:
            prompt = build_correction_prompt(task, current_code or "", failed_details, passed_indices)

        t_api_start = time.perf_counter()
        try:
            resp = client.interactions.create(
                model=model,
                input=prompt,
                generation_config={"thinking_level": thinking_level},
            )
            api_elapsed = time.perf_counter() - t_api_start
            total_api_time += api_elapsed

            if resp.usage:
                result.input_tokens += resp.usage.total_input_tokens or 0
                result.reasoning_tokens += resp.usage.total_thought_tokens or 0
                result.answer_tokens += resp.usage.total_output_tokens or 0
                result.total_tokens += resp.usage.total_tokens or 0

            code = parse_code_block(resp.output_text or "")
            if not code:
                result.error_message = "No Python code block found in response"
                if attempt < max_attempts:
                    time.sleep(delay_between_calls)
                    continue
                break

            current_code = code
            result.generated_code = code

            # Evaluate on training pairs in sandbox
            passed_indices = []
            failed_details = []

            for i, pair in enumerate(task["train"], 1):
                ok, pred_train, exec_time, msg = execute_transform_in_sandbox(
                    code, pair["input"], timeout_seconds=timeout_seconds
                )
                total_exec_time += exec_time

                if ok and compare_grids(pred_train, pair["output"]):
                    passed_indices.append(i)
                else:
                    failed_details.append({
                        "example_index": i,
                        "input": pair["input"],
                        "expected": pair["output"],
                        "actual": pred_train,
                        "error": msg if not ok else None,
                    })

            result.train_passed = len(passed_indices)

            # If all training examples pass, break from correction loop!
            if len(failed_details) == 0:
                if attempt > 1:
                    print(f"       [Loop Success] Passed all {train_total} training examples on attempt {attempt}!")
                break
            else:
                if attempt < max_attempts:
                    print(f"       [Correction Loop] Attempt {attempt} failed on {len(failed_details)}/{train_total} train examples. Requesting correction...")
                    time.sleep(delay_between_calls)

        except Exception as exc:
            total_api_time += (time.perf_counter() - t_api_start)
            result.error_message = str(exc)
            if attempt < max_attempts:
                time.sleep(delay_between_calls)
                continue
            break

    # Evaluate the final code on test inputs
    if current_code:
        test_passed = 0
        test_preds = []

        for pair in task["test"]:
            ok, pred_test, exec_time, msg = execute_transform_in_sandbox(
                current_code, pair["input"], timeout_seconds=timeout_seconds
            )
            total_exec_time += exec_time
            if ok and pred_test is not None:
                test_preds.append(pred_test)
                if compare_grids(pred_test, pair.get("output")):
                    test_passed += 1
            else:
                result.error_message = msg

        result.predictions = test_preds
        result.test_passed = test_passed
        result.solved = (test_passed == test_total and test_total > 0)
        result.grade = 1.0 if result.solved else 0.0

        # Check loop resolution and generalization gap
        if result.solved and result.attempts > 1:
            result.solved_with_loop = True

        train_perfect = (result.train_passed == train_total)
        if train_perfect and not result.solved:
            result.perfect_train_failed_test = True

    result.api_time_seconds = total_api_time
    result.exec_time_seconds = total_exec_time
    result.total_time_seconds = total_api_time + total_exec_time
    return result


# ==============================================================================
# Reporting & File Output
# ==============================================================================

def save_summary_json(summary: ExperimentSummary, output_path: pathlib.Path) -> None:
    """Saves the experiment summary to a structured JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)


def save_summary_txt(
    summaries: List[ExperimentSummary],
    output_path: pathlib.Path,
) -> None:
    """
    Saves a clean, human-readable text summary of the experiment(s).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 85)
    lines.append("EXPERIMENT SUMMARY: ARC-AGI-1 SOLVER EVALUATION")
    lines.append(f"Date & Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 85)

    for summary in summaries:
        lines.append(f"\nLabel:                   {summary.label}")
        lines.append(f"Model:                   {summary.model}")
        lines.append(f"Thinking Level:          {summary.thinking_level}")
        lines.append(f"Approach:                {summary.approach}")
        lines.append(f"Tasks Solved:            {summary.solved_tasks}/{summary.total_tasks} ({summary.accuracy_percentage:.1f}%)")
        if summary.approach == "code":
            lines.append(f"Solved with Loop:        {summary.solved_with_loop_count}")
            lines.append(f"Average Attempts/Task:   {summary.avg_attempts_per_task:.2f}")
            lines.append(f"Train Perfect / Test Fail: {summary.perfect_train_failed_test_count} {summary.perfect_train_failed_test_tasks}")
        lines.append(f"Input Tokens:            {summary.total_input_tokens:,}")
        lines.append(f"Reasoning Tokens:        {summary.total_reasoning_tokens:,}")
        lines.append(f"Answer Tokens:           {summary.total_answer_tokens:,}")
        lines.append(f"Total Tokens:            {summary.total_tokens:,}")
        lines.append(f"Total Time:              {summary.total_time_seconds:.2f}s (Avg: {summary.avg_time_per_task:.2f}s/task)")
        lines.append("-" * 85)
        lines.append(
            f"{'Task ID':<25} | {'Solved':<7} | {'Grade':<5} | {'Att':<4} | {'Train':<6} | {'Input':<7} | {'Reasoning':<9} | {'Time (s)':<8}"
        )
        lines.append("-" * 85)
        for t in summary.tasks:
            status = "YES" if t.solved else "NO"
            tr_str = f"{t.train_passed}/{t.train_total}"
            lines.append(
                f"{t.task_id:<25} | {status:<7} | {t.grade:<5.1f} | {t.attempts:<4} | {tr_str:<6} | {t.input_tokens:<7} | "
                f"{t.reasoning_tokens:<9} | {t.total_time_seconds:<8.2f}"
            )
        lines.append("-" * 85)

    # Comparison Table
    if len(summaries) > 1:
        lines.append("\n" + "=" * 85)
        lines.append("COMPARISON ACROSS APPROACHES")
        lines.append("=" * 85)
        lines.append(
            f"{'Approach':<15} | {'Solved':<10} | {'Loop Solved':<12} | {'Avg Attempts':<12} | {'Input Toks':<11} | {'Reason Toks':<12} | {'Total Time':<10}"
        )
        lines.append("-" * 85)
        for s in summaries:
            loop_str = f"{s.solved_with_loop_count}" if s.approach == "code" else "N/A"
            att_str = f"{s.avg_attempts_per_task:.2f}" if s.approach == "code" else "1.00"
            lines.append(
                f"{s.approach:<15} | {s.solved_tasks}/{s.total_tasks} ({s.accuracy_percentage:.0f}%) | "
                f"{loop_str:<12} | {att_str:<12} | {s.total_input_tokens:<11,} | {s.total_reasoning_tokens:<12,} | {s.total_time_seconds:<10.1f}s"
            )
        lines.append("=" * 85)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# Main Runner
# ==============================================================================

def run_experiment(
    client: genai.Client,
    tasks: List[Tuple[str, pathlib.Path]],
    approach: str,
    label: str,
    model: str = "gemini-3.5-flash-lite",
    thinking_level: str = "high",
    delay_between_calls: float = 1.0,
    timeout_seconds: float = 5.0,
    max_attempts: int = 3,
) -> ExperimentSummary:
    """Runs a complete evaluation pass for a given approach across all tasks."""
    print(f"\n{'='*75}")
    print(f" Starting Experiment [{approach.upper()}] - Label: {label}")
    print(f" Model: {model} | Reasoning: {thinking_level} | Tasks: {len(tasks)}")
    if approach == "code":
        print(f" Correction Loop: Enabled (max {max_attempts} attempts per task)")
    print(f"{'='*75}")

    summary = ExperimentSummary(
        label=label,
        model=model,
        thinking_level=thinking_level,
        approach=approach,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_tasks=len(tasks),
    )

    for idx, (rel_name, task_path) in enumerate(tasks, 1):
        task_data = load_task_json(task_path)
        print(f"\n[{idx:02d}/{len(tasks):02d}] Task: {rel_name}")

        if approach == "end_to_end":
            res = solve_task_end_to_end(
                client=client,
                model=model,
                thinking_level=thinking_level,
                task_id=rel_name,
                task=task_data,
            )
        else:
            res = solve_task_code_gen(
                client=client,
                model=model,
                thinking_level=thinking_level,
                task_id=rel_name,
                task=task_data,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                delay_between_calls=delay_between_calls,
            )

        summary.tasks.append(res)
        summary.total_input_tokens += res.input_tokens
        summary.total_reasoning_tokens += res.reasoning_tokens
        summary.total_answer_tokens += res.answer_tokens
        summary.total_tokens += res.total_tokens
        summary.total_time_seconds += res.total_time_seconds
        summary.total_attempts += res.attempts

        if res.solved:
            summary.solved_tasks += 1
        if res.solved_with_loop:
            summary.solved_with_loop_count += 1
        if res.perfect_train_failed_test:
            summary.perfect_train_failed_test_count += 1
            summary.perfect_train_failed_test_tasks.append(res.task_id)

        # Terminal status output required by prompt
        status_str = "CORRECT [PASS]" if res.solved else "INCORRECT [FAIL]"
        print(f"       Status:           {status_str} (Grade: {res.grade:.1f})")
        print(f"       Attempts Taken:   {res.attempts}")
        print(f"       Input Tokens:     {res.input_tokens:,}")
        print(f"       Reasoning Tokens: {res.reasoning_tokens:,}")
        print(f"       Answer Tokens:    {res.answer_tokens:,}")
        print(f"       Total Tokens:     {res.total_tokens:,}")
        if approach == "code":
            print(f"       Train Passed:     {res.train_passed}/{res.train_total}")
            print(f"       Solved via Loop:  {res.solved_with_loop}")
            print(f"       Train Pass/Test Fail: {res.perfect_train_failed_test}")
            print(f"       Code Exec Time:   {res.exec_time_seconds:.3f}s")
        print(f"       Execution Time:   {res.total_time_seconds:.2f}s (API: {res.api_time_seconds:.2f}s)")
        print(f"       Running Score:    {summary.solved_tasks}/{idx} ({summary.solved_tasks / idx * 100:.1f}%)")

        if res.error_message:
            print(f"       Note/Error:       {res.error_message}")

        if idx < len(tasks):
            time.sleep(delay_between_calls)

    summary.accuracy_percentage = (
        (summary.solved_tasks / summary.total_tasks * 100.0) if summary.total_tasks > 0 else 0.0
    )
    summary.avg_time_per_task = (
        (summary.total_time_seconds / summary.total_tasks) if summary.total_tasks > 0 else 0.0
    )
    summary.avg_attempts_per_task = (
        (summary.total_attempts / summary.total_tasks) if summary.total_tasks > 0 else 1.0
    )

    print(f"\n{'-'*75}")
    print(f" Completed Approach: {approach}")
    print(f" Solved: {summary.solved_tasks}/{summary.total_tasks} ({summary.accuracy_percentage:.1f}%)")
    if approach == "code":
        print(f" Solved with Correction Loop: {summary.solved_with_loop_count}")
        print(f" Average Attempts per Task:   {summary.avg_attempts_per_task:.2f}")
        print(f" Train Passed / Test Failed:  {summary.perfect_train_failed_test_count} ({summary.perfect_train_failed_test_tasks})")
    print(f" Total Tokens: {summary.total_tokens:,} | Total Time: {summary.total_time_seconds:.1f}s")
    print(f"{'-'*75}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Educational ARC-AGI-1 Solver with Gemini 3.5 Flash Lite"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to tarefas.txt (default: autodetected in workspace)",
    )
    parser.add_argument(
        "--mode",
        choices=["end_to_end", "code", "both"],
        default="both",
        help="Evaluation approach: 'end_to_end', 'code', or 'both' (default: 'both')",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="arc_agi1_gemini35_high",
        help="Experiment label for tracking and filenames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save summary files (default: ../resultados or ./resultados)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Gemini model identifier (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="high",
        help="Thinking/reasoning effort level (default: high)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Execution timeout in seconds for generated Python code (default: 5.0)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for code generation correction loop (default: 3)",
    )
    args = parser.parse_args()

    # Locate tasks file
    if args.tasks:
        tasks_file = pathlib.Path(args.tasks).resolve()
    else:
        candidates = [
            pathlib.Path.cwd() / "tarefas" / "tarefas.txt",
            pathlib.Path.cwd().parent / "tarefas" / "tarefas.txt",
            pathlib.Path(r"g:\Meu Drive\UFRGS\Aulas\IA002\mike_carvalho_borges\tarefas\tarefas.txt"),
        ]
        tasks_file = None
        for cand in candidates:
            if cand.exists():
                tasks_file = cand
                break
        if not tasks_file:
            print("Error: Could not locate tarefas.txt. Specify with --tasks <path>.")
            sys.exit(1)

    # Locate output directory
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir).resolve()
    else:
        candidates = [
            pathlib.Path.cwd() / "resultados",
            pathlib.Path.cwd().parent / "resultados",
            pathlib.Path(r"g:\Meu Drive\UFRGS\Aulas\IA002\mike_carvalho_borges\resultados"),
        ]
        output_dir = candidates[0]
        if pathlib.Path.cwd().name == "solver":
            output_dir = pathlib.Path.cwd().parent / "resultados"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Gemini Client...")
    client = setup_gemini_client()

    tasks = load_tasks_list(tasks_file)
    print(f"Loaded {len(tasks)} tasks from {tasks_file}")

    approaches_to_run = ["end_to_end", "code"] if args.mode == "both" else [args.mode]
    summaries = []

    for approach in approaches_to_run:
        exp_label = f"{args.label}_{approach}"
        summary = run_experiment(
            client=client,
            tasks=tasks,
            approach=approach,
            label=exp_label,
            model=args.model,
            thinking_level=args.thinking_level,
            delay_between_calls=args.delay,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
        )
        summaries.append(summary)

        # Save individual JSON summary
        json_path = output_dir / f"summary_{approach}.json"
        save_summary_json(summary, json_path)
        print(f"Saved {approach} summary to: {json_path}")

    # Save combined experiment_summary.txt and experiment_summary.json
    combined_json_path = output_dir / "experiment_summary.json"
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, ensure_ascii=False)

    summary_txt_path = output_dir / "experiment_summary.txt"
    save_summary_txt(summaries, summary_txt_path)
    print(f"\nFinal experiment text summary saved to: {summary_txt_path}")
    print(f"Combined JSON summary saved to:        {combined_json_path}")


if __name__ == "__main__":
    main()
