from __future__ import annotations

import json
import os
import pathlib
import shutil

import streamlit as st
from dotenv import load_dotenv

from eval_harness import EvalHarness
from git_manager import PromptGitManager
from loop import OptimizationLoop, RoundResult

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoEval Loop",
    layout="wide",
    page_icon="🔁",
)

load_dotenv()

BASE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Asset loader helpers
# ---------------------------------------------------------------------------

def _load_text(path: pathlib.Path) -> str | None:
    if not path.exists():
        st.error(f"Missing file: `{path}`")
        return None
    return path.read_text(encoding="utf-8").strip()


def _load_json(path: pathlib.Path) -> list | dict | None:
    if not path.exists():
        st.error(f"Missing file: `{path}`")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse `{path}`: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sub-renderers for Tab 1 setup columns
# ---------------------------------------------------------------------------

def _render_broken_prompt(broken_prompt: str) -> None:
    st.markdown("### The Broken Prompt")
    st.code(broken_prompt, language=None)
    st.caption(
        "**Why it fails:** vague intent handling (no instruction to identify what "
        "the user actually wants), no grounding instruction (may hallucinate policy "
        "details), and no honest-refusal rule (will attempt impossible requests "
        "rather than declining clearly)."
    )


def _render_criteria(criteria: list[dict]) -> None:
    st.markdown("### Eval Criteria (5 checks)")
    for c in criteria:
        with st.expander(f"**{c['id']}** — {c['label']}"):
            st.write(c["description"])
            st.markdown(
                f'<div style="background:#d4edda;color:#155724;padding:8px 12px;'
                f'border-radius:6px;margin-bottom:6px;">'
                f'<strong>✓ PASS</strong><br>{c["pass_example"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#f8d7da;color:#721c24;padding:8px 12px;'
                f'border-radius:6px;">'
                f'<strong>✗ FAIL</strong><br>{c["fail_example"]}</div>',
                unsafe_allow_html=True,
            )


def _render_scenarios(scenarios: list[dict]) -> None:
    st.markdown("### Test Scenarios (3 inputs)")
    for i, s in enumerate(scenarios, start=1):
        st.markdown(f"**{i}. {s['title']}**")
        st.markdown(
            f'<div style="background:#f5f5f5;color:#333;padding:10px 14px;'
            f'border-radius:6px;border-left:3px solid #ccc;margin-bottom:12px;">'
            f'<em>{s["user_message"]}</em></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Live progress feed renderer
# ---------------------------------------------------------------------------

def _render_progress_feed(
    placeholder: st.delta_generator.DeltaGenerator,
    results: list[RoundResult],
    max_rounds: int,
) -> None:
    with placeholder.container(border=True):
        st.caption(f"**Live progress** — {len(results)} / {max_rounds} rounds complete")
        for r in results:
            kept = r.kept
            color = "#155724" if kept else "#721c24"
            bg = "#d4edda" if kept else "#f8d7da"
            icon = "✓ KEPT" if kept else "✗ REVERTED"
            delta = r.score_after - r.score_before
            sign = "+" if delta >= 0 else ""
            criterion_line = "  ".join(
                f"{cid}: {v:.0f}%" for cid, v in r.per_criterion_scores.items()
            )
            st.markdown(
                f'<div style="background:{bg};color:{color};padding:7px 12px;'
                f'border-radius:5px;margin-bottom:5px;font-size:0.9em;">'
                f'<strong>Round {r.round_num}</strong> &nbsp;|&nbsp; '
                f'{r.score_before:.1f}% → <strong>{r.score_after:.1f}%</strong> '
                f'({sign}{delta:.1f}%) &nbsp;|&nbsp; {r.change_description} '
                f'&nbsp;|&nbsp; <strong>{icon}</strong><br>'
                f'<span style="font-size:0.85em;opacity:0.85;">'
                f'Criteria: {criterion_line}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 1: Try the Demo
# ---------------------------------------------------------------------------

def tab_demo() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("AutoEval Loop")
    st.markdown(
        "**Autonomous agent prompt optimizer.** "
        "Inspired by Karpathy's autoresearch."
    )
    st.divider()

    # ── Load demo assets ────────────────────────────────────────────────────
    broken_prompt = _load_text(BASE / "demo" / "broken_prompt.txt")
    criteria = _load_json(BASE / "demo" / "eval_criteria.json")
    scenarios = _load_json(BASE / "demo" / "test_scenarios.json")

    if broken_prompt is None or criteria is None or scenarios is None:
        st.stop()

    # ── Three-column setup display ───────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        _render_broken_prompt(broken_prompt)
    with col2:
        _render_criteria(criteria)
    with col3:
        _render_scenarios(scenarios)

    st.divider()

    # ── Settings ────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        max_rounds = st.slider(
            "Number of rounds",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="Maximum rounds the loop will run before stopping.",
        )
        quick_mode = st.checkbox(
            "Quick mode (2 responses per scenario, faster but less accurate)",
            value=True,
            help=(
                "Uses 2 agent responses per scenario instead of 5. "
                "Reduces API calls by 60% — good for demos and development."
            ),
        )
    # Provide defaults when expander hasn't been interacted with
    if "max_rounds" not in st.session_state:
        st.session_state.max_rounds = 20
    if "quick_mode" not in st.session_state:
        st.session_state.quick_mode = True

    # ── Run button ───────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_clicked = st.button(
            "▶ Run Optimization Loop",
            use_container_width=True,
            type="primary",
        )

    # ── Execution ────────────────────────────────────────────────────────────
    if run_clicked:
        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not groq_key or groq_key == "your_groq_api_key_here":
            st.error("GROQ_API_KEY is not set. Add it to your `.env` file.")
            st.stop()
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            st.error("GEMINI_API_KEY is not set. Add it to your `.env` file.")
            st.stop()

        # Reset session state for a fresh run
        st.session_state.round_results = []
        st.session_state.summary = None

        # Clean workspace so every demo run starts from scratch
        demo_workspace = BASE / "workspace"
        git_dir = demo_workspace / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
        for leftover in ("current_prompt.txt", "results.log"):
            (demo_workspace / leftover).unlink(missing_ok=True)

        # Build engine objects
        harness = EvalHarness(
            system_prompt=broken_prompt,
            criteria=criteria,
            scenarios=scenarios,
            groq_api_key=groq_key,
            gemini_api_key=gemini_key,
        )
        if quick_mode:
            harness.run_evaluation = harness.run_quick_evaluation  # type: ignore[method-assign]

        git_mgr = PromptGitManager(workspace_path=str(demo_workspace))

        loop = OptimizationLoop(
            eval_harness=harness,
            git_manager=git_mgr,
            gemini_api_key=gemini_key,
            max_rounds=max_rounds,
            target_score=95.0,
            consecutive_target_rounds=3,
        )

        # Progress bar + live feed placeholders
        progress_bar = st.progress(0.0, text="Starting…")
        feed_placeholder = st.empty()

        def on_round(result: RoundResult) -> None:
            st.session_state.round_results.append(result)
            done = len(st.session_state.round_results)
            pct = done / max_rounds
            label = (
                f"Round {result.round_num} / {max_rounds} — "
                f"score {result.score_after:.1f}%"
            )
            progress_bar.progress(min(pct, 1.0), text=label)
            _render_progress_feed(feed_placeholder, st.session_state.round_results, max_rounds)

        with st.spinner("Optimization running… this may take a few minutes."):
            summary = loop.run(
                initial_prompt=broken_prompt,
                on_round_complete=on_round,
            )

        st.session_state.summary = summary
        progress_bar.progress(1.0, text="Complete!")

    # ── Summary (shown after run or from session state) ───────────────────
    if st.session_state.get("summary"):
        summary = st.session_state.summary
        st.divider()
        st.markdown("### Results")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Initial Score", f"{summary['initial_score']:.1f}%")
        m2.metric(
            "Final Score",
            f"{summary['final_score']:.1f}%",
            delta=f"+{summary['improvement']:.1f}%",
        )
        m3.metric("Rounds Run", summary["rounds_run"])
        m4.metric("Rounds Kept", summary["rounds_kept"])
        m5.metric("Rounds Reverted", summary["rounds_reverted"])

        st.markdown("#### Final Prompt")
        st.code(summary["final_prompt"], language=None)

        with st.expander("View prompt diff (original vs. final)"):
            diff_text = summary.get("diff", "")
            if not diff_text:
                # Generate diff on the fly if not in summary
                git_mgr_r = PromptGitManager(workspace_path=str(BASE / "workspace"))
                try:
                    diff_text = git_mgr_r.get_prompt_diff(summary["original_prompt"])
                except Exception:
                    diff_text = ""
            if diff_text:
                st.code(diff_text, language="diff")
            else:
                st.info("No changes — prompt is identical to original.")

        with st.expander("View experiment log"):
            st.text(summary["full_log"])

        with st.expander("View commit history"):
            for entry in reversed(summary["commit_history"]):
                r_label = (
                    f"Round {entry['round']}" if entry["round"] else "Baseline"
                )
                score_label = (
                    f"{entry['score']:.1f}%" if entry["score"] is not None else "—"
                )
                st.markdown(
                    f"- `{r_label}` &nbsp; score: **{score_label}** &nbsp; "
                    f"_{entry['message']}_"
                )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Try the Demo", "Optimize Your Own"])

with tab1:
    tab_demo()

with tab2:
    st.info("Coming soon.")
