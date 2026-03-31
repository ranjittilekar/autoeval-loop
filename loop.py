from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import google.generativeai as genai

from eval_harness import EvalHarness, EvalResult
from git_manager import PromptGitManager


# ---------------------------------------------------------------------------
# RoundResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_num: int
    score_before: float
    score_after: float
    change_description: str
    kept: bool
    per_criterion_scores: dict[str, float]   # criterion_id → pass_rate after this round
    timestamp: str


# ---------------------------------------------------------------------------
# OptimizationLoop
# ---------------------------------------------------------------------------

class OptimizationLoop:
    GEMINI_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        eval_harness: EvalHarness,
        git_manager: PromptGitManager,
        gemini_api_key: str,
        max_rounds: int = 50,
        target_score: float = 95.0,
        consecutive_target_rounds: int = 3,
    ) -> None:
        self.eval_harness = eval_harness
        self.git_manager = git_manager
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.consecutive_target_rounds = consecutive_target_rounds

        genai.configure(api_key=gemini_api_key)
        self._gemini = genai.GenerativeModel(self.GEMINI_MODEL)

        # State populated during run()
        self._original_prompt: str = ""
        self._initial_score: float = 0.0
        self._final_score: float = 0.0
        self._rounds_run: int = 0
        self._rounds_kept: int = 0
        self._rounds_reverted: int = 0

    # ------------------------------------------------------------------
    # Public: run the loop
    # ------------------------------------------------------------------

    def run(
        self,
        initial_prompt: str,
        on_round_complete: Callable[[RoundResult], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run the optimization loop. Calls on_round_complete(RoundResult) after
        each round if provided (used to stream progress to Streamlit).
        Returns get_summary() when done.
        """
        self._original_prompt = initial_prompt
        self._rounds_run = 0
        self._rounds_kept = 0
        self._rounds_reverted = 0
        consecutive_at_target = 0

        # Init workspace with the starting prompt
        self.git_manager.init_workspace(initial_prompt)
        self.git_manager.write_log_entry(
            f"Run started. max_rounds={self.max_rounds}, target={self.target_score}%"
        )

        # Baseline score (round 0)
        self.eval_harness.system_prompt = initial_prompt
        baseline_eval = self.eval_harness.run_evaluation()
        self._initial_score = baseline_eval.overall_score
        self._final_score = baseline_eval.overall_score
        self.git_manager.write_log_entry(
            f"BASELINE score: {self._initial_score:.1f}%  "
            f"{self._fmt_criterion_scores(baseline_eval.per_criterion_scores)}"
        )

        for round_num in range(1, self.max_rounds + 1):
            self._rounds_run += 1

            # ── Step 1-2: read current prompt and score it ──────────────
            current_prompt = self.git_manager.read_current_prompt()
            self.eval_harness.system_prompt = current_prompt
            current_eval = self.eval_harness.run_evaluation()
            score_before = current_eval.overall_score

            # ── Step 3: identify worst criterion ────────────────────────
            worst_criterion = self._find_worst_criterion(current_eval)

            # ── Step 4: generate one targeted edit ──────────────────────
            candidate_prompt = self._generate_prompt_edit(
                current_prompt, worst_criterion, current_eval
            )

            # ── Step 5-6: write candidate and evaluate it ────────────────
            self.git_manager.write_candidate_prompt(candidate_prompt)
            self.eval_harness.system_prompt = candidate_prompt
            candidate_eval = self.eval_harness.run_evaluation()
            score_after = candidate_eval.overall_score

            change_description = (
                f"targeted fix for {worst_criterion['label']} "
                f"(was {current_eval.per_criterion_scores[worst_criterion['id']]:.0f}%)"
            )

            # ── Step 7-8: commit (tentative) then keep or revert ─────────
            self.git_manager.commit_round(round_num, score_after, change_description)

            if score_after > score_before:
                kept = True
                self._rounds_kept += 1
                self._final_score = score_after
                self.git_manager.write_log_entry(
                    f"ROUND {round_num}: KEPT   "
                    f"{score_before:.1f}% → {score_after:.1f}%  "
                    f"(+{score_after - score_before:.1f}%)  — {change_description}"
                )
            else:
                kept = False
                self._rounds_reverted += 1
                self.git_manager.revert_round(
                    round_num,
                    score_after,
                    f"{change_description} — score did not improve "
                    f"({score_before:.1f}% → {score_after:.1f}%)",
                )

            # ── Step 9: build result and fire callback ───────────────────
            round_result = RoundResult(
                round_num=round_num,
                score_before=score_before,
                score_after=score_after,
                change_description=change_description,
                kept=kept,
                per_criterion_scores=candidate_eval.per_criterion_scores,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            if on_round_complete is not None:
                on_round_complete(round_result)

            # ── Step 10: check stopping conditions ───────────────────────
            if kept and score_after >= self.target_score:
                consecutive_at_target += 1
                if consecutive_at_target >= self.consecutive_target_rounds:
                    self.git_manager.write_log_entry(
                        f"STOPPING: target {self.target_score}% reached for "
                        f"{self.consecutive_target_rounds} consecutive rounds."
                    )
                    break
            else:
                consecutive_at_target = 0

        self.git_manager.write_log_entry(
            f"Run complete. rounds={self._rounds_run}  "
            f"kept={self._rounds_kept}  reverted={self._rounds_reverted}  "
            f"final={self._final_score:.1f}%"
        )

        return self.get_summary()

    # ------------------------------------------------------------------
    # Public: summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        final_prompt = self.git_manager.read_current_prompt()
        return {
            "initial_score": self._initial_score,
            "final_score": self._final_score,
            "improvement": round(self._final_score - self._initial_score, 1),
            "rounds_run": self._rounds_run,
            "rounds_kept": self._rounds_kept,
            "rounds_reverted": self._rounds_reverted,
            "original_prompt": self._original_prompt,
            "final_prompt": final_prompt,
            "full_log": self.git_manager.get_full_log(),
            "commit_history": self.git_manager.get_commit_history(),
        }

    # ------------------------------------------------------------------
    # Private: prompt editing
    # ------------------------------------------------------------------

    def _find_worst_criterion(self, eval_result: EvalResult) -> dict:
        worst_id = min(
            eval_result.per_criterion_scores,
            key=lambda cid: eval_result.per_criterion_scores[cid],
        )
        return next(
            c for c in self.eval_harness.criteria if c["id"] == worst_id
        )

    def _generate_prompt_edit(
        self,
        current_prompt: str,
        worst_criterion: dict,
        eval_result: EvalResult,
    ) -> str:
        criterion_block = (
            f"ID: {worst_criterion['id']}\n"
            f"Label: {worst_criterion['label']}\n"
            f"Description: {worst_criterion['description']}\n"
            f"PASS example: {worst_criterion['pass_example']}\n"
            f"FAIL example: {worst_criterion['fail_example']}\n"
            f"Current pass rate: "
            f"{eval_result.per_criterion_scores[worst_criterion['id']]:.1f}%"
        )

        prompt = (
            f"You are a prompt engineer. The system prompt below is failing on "
            f"this specific criterion:\n\n{criterion_block}\n\n"
            f"Make ONE targeted change to fix this. Do not change anything else. "
            f"Return only the complete updated system prompt with no explanation.\n\n"
            f"SYSTEM PROMPT:\n{current_prompt}"
        )

        return self._call_gemini(prompt).strip()

    # ------------------------------------------------------------------
    # Private: Gemini call with exponential backoff
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._gemini.generate_content(prompt)
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(f"  [Loop/Gemini] Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError(
            f"Gemini call failed after 3 attempts: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_criterion_scores(scores: dict[str, float]) -> str:
        return "  ".join(f"{cid}:{v:.0f}%" for cid, v in scores.items())


# ---------------------------------------------------------------------------
# __main__ — full demo run with live per-round output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    import pathlib
    import shutil
    from dotenv import load_dotenv

    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not groq_key or not gemini_key:
        raise SystemExit("ERROR: Set GROQ_API_KEY and GEMINI_API_KEY in .env")

    base = pathlib.Path(__file__).parent

    initial_prompt = (base / "demo" / "broken_prompt.txt").read_text().strip()
    criteria = json.loads((base / "demo" / "eval_criteria.json").read_text())
    scenarios = json.loads((base / "demo" / "test_scenarios.json").read_text())

    # Clean workspace before demo run
    demo_workspace = base / "workspace"
    shutil.rmtree(demo_workspace / ".git", ignore_errors=True)
    (demo_workspace / "current_prompt.txt").unlink(missing_ok=True)
    (demo_workspace / "results.log").unlink(missing_ok=True)

    harness = EvalHarness(
        system_prompt=initial_prompt,
        criteria=criteria,
        scenarios=scenarios,
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
    )
    git_mgr = PromptGitManager(workspace_path=str(demo_workspace))

    loop = OptimizationLoop(
        eval_harness=harness,
        git_manager=git_mgr,
        gemini_api_key=gemini_key,
        max_rounds=10,           # keep short for demo; full run uses 50
        target_score=95.0,
        consecutive_target_rounds=3,
    )

    # Live callback printed as each round completes
    def on_round(result: RoundResult) -> None:
        icon = "KEPT    " if result.kept else "REVERTED"
        delta = result.score_after - result.score_before
        sign = "+" if delta >= 0 else ""
        print(
            f"  Round {result.round_num:>2} [{icon}]  "
            f"{result.score_before:.1f}% → {result.score_after:.1f}%  "
            f"({sign}{delta:.1f}%)  |  {result.change_description}"
        )
        criterion_line = "  ".join(
            f"{cid}:{v:.0f}%" for cid, v in result.per_criterion_scores.items()
        )
        print(f"            Criteria: {criterion_line}")

    print("=" * 70)
    print("AutoEval Loop — Optimization Run (max 10 rounds for demo)")
    print(f"Starting prompt: '{initial_prompt[:70]}…'")
    print("=" * 70)

    summary = loop.run(initial_prompt=initial_prompt, on_round_complete=on_round)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Initial score : {summary['initial_score']:.1f}%")
    print(f"  Final score   : {summary['final_score']:.1f}%")
    print(f"  Improvement   : +{summary['improvement']:.1f}%")
    print(f"  Rounds run    : {summary['rounds_run']}")
    print(f"  Rounds kept   : {summary['rounds_kept']}")
    print(f"  Rounds reverted: {summary['rounds_reverted']}")
    print()
    print("COMMIT HISTORY")
    print("-" * 70)
    for entry in reversed(summary["commit_history"]):
        r = f"round {entry['round']}" if entry["round"] is not None else "baseline"
        s = f"{entry['score']:.1f}%" if entry["score"] is not None else "—"
        print(f"  {r:<10}  score={s:<8}  {entry['message'][:55]}")
    print()
    print("FINAL PROMPT")
    print("-" * 70)
    print(summary["final_prompt"])
    print()
    print("FULL LOG")
    print("-" * 70)
    print(summary["full_log"])
    print("=" * 70)
