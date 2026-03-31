from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import google.generativeai as genai
from groq import Groq


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    overall_score: float                          # 0-100
    per_criterion_scores: dict[str, float]        # criterion_id -> pass_rate (0-100)
    per_scenario_scores: dict[str, float]         # scenario_id  -> pass_rate (0-100)
    raw_results: list[dict[str, Any]]             # one entry per (scenario, response_idx, criterion)
    timestamp: str
    responses_per_scenario: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EvalHarness:
    GROQ_MODEL = "llama-3.1-8b-instant"
    GEMINI_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        system_prompt: str,
        criteria: list[dict],
        scenarios: list[dict],
        groq_api_key: str,
        gemini_api_key: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.criteria = criteria
        self.scenarios = scenarios

        self.groq_client = Groq(api_key=groq_api_key)

        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.GEMINI_MODEL)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_evaluation(self) -> EvalResult:
        """Full evaluation: 5 responses per scenario."""
        return self._evaluate(responses_per_scenario=5)

    def run_quick_evaluation(self) -> EvalResult:
        """Fast evaluation for development: 2 responses per scenario."""
        return self._evaluate(responses_per_scenario=2)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _evaluate(self, responses_per_scenario: int) -> EvalResult:
        raw_results: list[dict[str, Any]] = []

        for scenario in self.scenarios:
            responses = self._generate_responses(scenario, responses_per_scenario)
            for idx, response_text in enumerate(responses):
                for criterion in self.criteria:
                    judgment = self._judge_response(response_text, criterion, scenario)
                    raw_results.append({
                        "scenario_id": scenario["id"],
                        "scenario_title": scenario["title"],
                        "response_idx": idx,
                        "criterion_id": criterion["id"],
                        "criterion_label": criterion["label"],
                        "response_text": response_text,
                        "result": judgment["result"],
                        "reason": judgment["reason"],
                    })

        return self._aggregate(raw_results, responses_per_scenario)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_responses(
        self, scenario: dict, count: int
    ) -> list[str]:
        responses = []
        for _ in range(count):
            text = self._call_groq(scenario["user_message"])
            responses.append(text)
        return responses

    def _call_groq(self, user_message: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.7,
                )
                return completion.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(f"  [Groq] Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError(f"Groq call failed after 3 attempts: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    def _judge_response(self, response_text: str, criterion: dict, scenario: dict) -> dict:
        prompt = self._build_judge_prompt(response_text, criterion, scenario)
        raw = self._call_gemini(prompt)
        return self._parse_judgment(raw, criterion["id"])

    def _build_judge_prompt(
        self, response_text: str, criterion: dict, scenario: dict
    ) -> str:
        return f"""You are a strict evaluator. Answer only with valid JSON. Do not add explanation outside the JSON object.

CRITERION
ID: {criterion["id"]}
Label: {criterion["label"]}
Description: {criterion["description"]}

PASS example: {criterion["pass_example"]}
FAIL example: {criterion["fail_example"]}

USER MESSAGE (the message the agent was responding to):
{scenario["user_message"]}

AGENT RESPONSE TO EVALUATE:
{response_text}

Evaluate whether the agent response PASSES or FAILS the criterion above.
Respond with exactly this JSON structure:
{{"result": "PASS", "reason": "one sentence explaining your judgment"}}
or
{{"result": "FAIL", "reason": "one sentence explaining your judgment"}}"""

    def _call_gemini(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(f"  [Gemini] Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after 3 attempts: {last_exc}") from last_exc

    def _parse_judgment(self, raw: str, criterion_id: str) -> dict:
        """Extract JSON from Gemini response, tolerating markdown code fences."""
        text = raw.strip()
        # Strip ```json ... ``` fences if present
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Grab the first {...} block in the response
            brace_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        try:
            data = json.loads(text)
            result = data.get("result", "").upper()
            if result not in ("PASS", "FAIL"):
                result = "FAIL"
            return {"result": result, "reason": data.get("reason", "")}
        except json.JSONDecodeError:
            print(f"  [Judge] Could not parse JSON for {criterion_id}. Raw: {raw[:120]}")
            return {"result": "FAIL", "reason": "parse error — defaulted to FAIL"}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, raw_results: list[dict], responses_per_scenario: int
    ) -> EvalResult:
        # Per-criterion pass rates
        per_criterion: dict[str, list[bool]] = {c["id"]: [] for c in self.criteria}
        for entry in raw_results:
            per_criterion[entry["criterion_id"]].append(entry["result"] == "PASS")

        per_criterion_scores = {
            cid: round(100 * sum(passes) / len(passes), 1) if passes else 0.0
            for cid, passes in per_criterion.items()
        }

        # Per-scenario pass rates (across all criteria for that scenario)
        per_scenario: dict[str, list[bool]] = {s["id"]: [] for s in self.scenarios}
        for entry in raw_results:
            per_scenario[entry["scenario_id"]].append(entry["result"] == "PASS")

        per_scenario_scores = {
            sid: round(100 * sum(passes) / len(passes), 1) if passes else 0.0
            for sid, passes in per_scenario.items()
        }

        # Overall score: mean of per-criterion pass rates
        overall = round(
            sum(per_criterion_scores.values()) / len(per_criterion_scores)
            if per_criterion_scores else 0.0,
            1,
        )

        return EvalResult(
            overall_score=overall,
            per_criterion_scores=per_criterion_scores,
            per_scenario_scores=per_scenario_scores,
            raw_results=raw_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
            responses_per_scenario=responses_per_scenario,
        )


# ---------------------------------------------------------------------------
# __main__ — smoke test against demo assets
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pathlib
    from dotenv import load_dotenv

    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not groq_key or not gemini_key:
        raise SystemExit("ERROR: Set GROQ_API_KEY and GEMINI_API_KEY in .env")

    base = pathlib.Path(__file__).parent

    system_prompt = (base / "demo" / "broken_prompt.txt").read_text().strip()
    criteria = json.loads((base / "demo" / "eval_criteria.json").read_text())
    scenarios = json.loads((base / "demo" / "test_scenarios.json").read_text())

    harness = EvalHarness(
        system_prompt=system_prompt,
        criteria=criteria,
        scenarios=scenarios,
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
    )

    print("Running quick evaluation (2 responses/scenario × 3 scenarios × 5 criteria = 30 judgments)…\n")
    result = harness.run_quick_evaluation()

    print("=" * 60)
    print(f"OVERALL SCORE: {result.overall_score:.1f}%")
    print(f"Timestamp:     {result.timestamp}")
    print(f"Responses/scenario: {result.responses_per_scenario}")
    print()

    print("PER-CRITERION SCORES")
    print("-" * 40)
    for cid, score in result.per_criterion_scores.items():
        label = next(c["label"] for c in criteria if c["id"] == cid)
        bar = "#" * int(score / 5)
        print(f"  {cid} {label:<38} {score:5.1f}%  {bar}")
    print()

    print("PER-SCENARIO SCORES")
    print("-" * 40)
    for sid, score in result.per_scenario_scores.items():
        title = next(s["title"] for s in scenarios if s["id"] == sid)
        print(f"  {sid} {title:<38} {score:5.1f}%")
    print()

    print("SAMPLE RAW RESULTS (first 6)")
    print("-" * 40)
    for entry in result.raw_results[:6]:
        icon = "PASS" if entry["result"] == "PASS" else "FAIL"
        print(f"  [{icon}] {entry['scenario_id']} × {entry['criterion_id']}  — {entry['reason']}")
    print("=" * 60)
