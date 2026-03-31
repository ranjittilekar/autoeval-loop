from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import zipfile

import pandas as pd
import plotly.graph_objects as go
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
# Setup column renderers
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
# Live progress feed
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
# Results tab builders
# ---------------------------------------------------------------------------

def _build_score_chart(
    round_results: list[RoundResult],
    initial_score: float,
    target_score: float = 95.0,
) -> go.Figure:
    kept_x, kept_y = [0], [initial_score]
    rev_x: list[int] = []
    rev_y: list[float] = []

    for r in round_results:
        if r.kept:
            kept_x.append(r.round_num)
            kept_y.append(r.score_after)
        else:
            rev_x.append(r.round_num)
            rev_y.append(r.score_after)

    total_rounds = round_results[-1].round_num if round_results else 0

    fig = go.Figure()

    # KEPT trace — green filled circles
    fig.add_trace(go.Scatter(
        x=kept_x,
        y=kept_y,
        mode="lines+markers",
        name="Kept",
        line=dict(color="#2ca02c", width=2),
        marker=dict(symbol="circle", size=10, color="#2ca02c"),
    ))

    # REVERTED trace — red X markers (no connecting line)
    if rev_x:
        fig.add_trace(go.Scatter(
            x=rev_x,
            y=rev_y,
            mode="markers",
            name="Reverted",
            marker=dict(symbol="x", size=12, color="#d62728", line=dict(width=2)),
        ))

    # Target line
    x_max = max(total_rounds, 1)
    fig.add_shape(
        type="line",
        x0=0, x1=x_max,
        y0=target_score, y1=target_score,
        line=dict(color="#ff7f0e", dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=x_max, y=target_score,
        text=f"Target ({target_score:.0f}%)",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(color="#ff7f0e", size=11),
    )

    # Annotate first and last kept points
    if kept_x and kept_y:
        fig.add_annotation(
            x=kept_x[0], y=kept_y[0],
            text=f"{kept_y[0]:.1f}%",
            showarrow=True, arrowhead=2, ay=-30,
            font=dict(size=11),
        )
        if len(kept_x) > 1:
            fig.add_annotation(
                x=kept_x[-1], y=kept_y[-1],
                text=f"{kept_y[-1]:.1f}%",
                showarrow=True, arrowhead=2, ay=-30,
                font=dict(size=11),
            )

    fig.update_layout(
        title=f"Score Progression: Round 0 → Round {total_rounds}",
        xaxis_title="Round",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=60, l=50, r=30),
        height=420,
    )

    return fig


def _build_heatmap(
    round_results: list[RoundResult],
    criteria: list[dict],
) -> go.Figure:
    cids = [c["id"] for c in criteria]
    short_labels = [f"{c['id']}: {c['label'][:20]}" for c in criteria]
    round_nums = [r.round_num for r in round_results]

    # Build matrix: rows = criteria, cols = rounds
    z = [
        [r.per_criterion_scores.get(cid, 0) for r in round_results]
        for cid in cids
    ]
    text = [
        [f"{v:.0f}%" for v in row]
        for row in z
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"R{n}" for n in round_nums],
        y=short_labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#d62728"],
            [0.5, "#ffdd57"],
            [1.0, "#2ca02c"],
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(title="Pass rate %"),
    ))

    fig.update_layout(
        title="Per-Criterion Pass Rate by Round",
        xaxis_title="Round",
        yaxis_title="Criterion",
        height=320,
        margin=dict(t=50, b=50, l=200, r=30),
    )

    return fig


def _build_summary_df(
    criteria: list[dict],
    round_results: list[RoundResult],
    baseline_criterion_scores: dict[str, float],
) -> pd.DataFrame:
    if not round_results:
        return pd.DataFrame()

    final_scores = round_results[-1].per_criterion_scores
    rows = []
    for c in criteria:
        cid = c["id"]
        baseline = round(baseline_criterion_scores.get(cid, 0.0), 1)
        final = round(final_scores.get(cid, 0.0), 1)
        rows.append({
            "Criterion": f"{cid}: {c['label']}",
            "Baseline %": baseline,
            "Final %": final,
            "Change (pp)": round(final - baseline, 1),
        })

    return pd.DataFrame(rows)


def _highlight_positive_change(row: pd.Series) -> list[str]:
    color = "background-color: #d4edda" if row["Change (pp)"] > 0 else ""
    return [color] * len(row)


def _build_download_zip(summary: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("final_prompt.txt", summary.get("final_prompt", ""))
        zf.writestr("results.log", summary.get("full_log", ""))

        summary_export = {
            k: v for k, v in summary.items()
            if k not in ("full_log", "commit_history")  # already separate files
        }
        # commit_history is small — include it
        summary_export["commit_history"] = summary.get("commit_history", [])
        zf.writestr("summary.json", json.dumps(summary_export, indent=2))

    return buf.getvalue()


def _render_results_tabs(
    summary: dict,
    round_results: list[RoundResult],
    criteria: list[dict],
    baseline_criterion_scores: dict[str, float],
) -> None:
    score_tab, heatmap_tab, diff_tab = st.tabs(
        ["Score Chart", "Criterion Breakdown", "Prompt Diff & Log"]
    )

    # ── Score Chart ──────────────────────────────────────────────────────────
    with score_tab:
        if round_results:
            fig = _build_score_chart(
                round_results,
                initial_score=summary["initial_score"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No round results to chart yet.")

    # ── Criterion Breakdown ───────────────────────────────────────────────────
    with heatmap_tab:
        if round_results:
            heatmap_fig = _build_heatmap(round_results, criteria)
            st.plotly_chart(heatmap_fig, use_container_width=True)

            st.markdown("#### Summary by Criterion")
            df = _build_summary_df(criteria, round_results, baseline_criterion_scores)
            if not df.empty:
                styled = df.style.apply(_highlight_positive_change, axis=1).format(
                    {"Baseline %": "{:.1f}", "Final %": "{:.1f}", "Change (pp)": "{:+.1f}"}
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No round results to display yet.")

    # ── Prompt Diff & Log ────────────────────────────────────────────────────
    with diff_tab:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Original Prompt**")
            st.code(summary.get("original_prompt", ""), language=None)
        with right:
            st.markdown("**Optimized Prompt**")
            st.code(summary.get("final_prompt", ""), language=None)

        st.markdown("#### What Changed")
        history = summary.get("commit_history", [])
        kept_rounds = [
            h for h in reversed(history)
            if h.get("round") and h["round"] > 0
        ]
        if kept_rounds:
            for i, h in enumerate(kept_rounds, start=1):
                score_str = f"{h['score']:.1f}%" if h["score"] is not None else "—"
                st.markdown(f"{i}. Round {h['round']}: _{h['message']}_ (score: {score_str})")
        else:
            st.info("No rounds were kept — prompt is unchanged.")

        st.markdown("#### Full Experiment Log")
        st.text_area(
            label="Full Experiment Log",
            value=summary.get("full_log", ""),
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    # ── Download button ───────────────────────────────────────────────────────
    st.divider()
    zip_bytes = _build_download_zip(summary)
    st.download_button(
        label="⬇ Download Results (prompt + log + summary)",
        data=zip_bytes,
        file_name="autoeval_results.zip",
        mime="application/zip",
        use_container_width=True,
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

    # ── Load demo assets ─────────────────────────────────────────────────────
    broken_prompt = _load_text(BASE / "demo" / "broken_prompt.txt")
    criteria = _load_json(BASE / "demo" / "eval_criteria.json")
    scenarios = _load_json(BASE / "demo" / "test_scenarios.json")

    if broken_prompt is None or criteria is None or scenarios is None:
        st.stop()

    # ── Three-column setup display ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        _render_broken_prompt(broken_prompt)
    with col2:
        _render_criteria(criteria)
    with col3:
        _render_scenarios(scenarios)

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
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
    if "max_rounds" not in st.session_state:
        st.session_state.max_rounds = 20
    if "quick_mode" not in st.session_state:
        st.session_state.quick_mode = True

    # ── Run button ────────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_clicked = st.button(
            "▶ Run Optimization Loop",
            use_container_width=True,
            type="primary",
        )

    # ── Execution ─────────────────────────────────────────────────────────────
    if run_clicked:
        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not groq_key or groq_key == "your_groq_api_key_here":
            st.error("GROQ_API_KEY is not set. Add it to your `.env` file.")
            st.stop()
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            st.error("GEMINI_API_KEY is not set. Add it to your `.env` file.")
            st.stop()

        st.session_state.round_results = []
        st.session_state.summary = None
        st.session_state.baseline_criterion_scores = {}

        # Clean workspace for a fresh run
        demo_workspace = BASE / "workspace"
        if (demo_workspace / ".git").exists():
            shutil.rmtree(demo_workspace / ".git")
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

        # One pre-loop baseline eval to capture per-criterion baseline scores
        # (loop.run() also runs one internally, so this is a deliberate extra call
        # purely to populate the "Baseline %" column in the summary table)
        with st.spinner("Scoring baseline prompt…"):
            baseline_eval = harness.run_evaluation()
        st.session_state.baseline_criterion_scores = baseline_eval.per_criterion_scores

        git_mgr = PromptGitManager(workspace_path=str(demo_workspace))

        loop = OptimizationLoop(
            eval_harness=harness,
            git_manager=git_mgr,
            gemini_api_key=gemini_key,
            max_rounds=max_rounds,
            target_score=95.0,
            consecutive_target_rounds=3,
        )

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
            _render_progress_feed(
                feed_placeholder, st.session_state.round_results, max_rounds
            )

        with st.spinner("Optimization running… this may take a few minutes."):
            summary = loop.run(
                initial_prompt=broken_prompt,
                on_round_complete=on_round,
            )

        st.session_state.summary = summary
        progress_bar.progress(1.0, text="Complete!")

    # ── Metrics + results panels (shown after run or from session state) ───────
    if st.session_state.get("summary"):
        summary = st.session_state.summary
        round_results: list[RoundResult] = st.session_state.get("round_results", [])
        baseline_criterion_scores: dict = st.session_state.get(
            "baseline_criterion_scores", {}
        )

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

        st.markdown("")  # spacing
        _render_results_tabs(
            summary=summary,
            round_results=round_results,
            criteria=criteria,
            baseline_criterion_scores=baseline_criterion_scores,
        )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Try the Demo", "Optimize Your Own"])

with tab1:
    tab_demo()

with tab2:
    st.info("Coming soon.")
