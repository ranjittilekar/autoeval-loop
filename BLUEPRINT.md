# BLUEPRINT.md
# AutoEval Loop: Autonomous Agent Prompt Optimizer

> A PM-built tool that runs 50+ scored experiments overnight to systematically
> improve AI agent system prompts. Inspired by Karpathy's autoresearch pattern.

---

## Why This Project Exists

Every company shipping an AI agent faces the same problem: the system prompt is the
single biggest lever on output quality, but most teams tune it by running 5 manual
test conversations and adjusting based on gut feel. There is no scoring. No experiment
log. No way to prove the new prompt is actually better.

This tool applies Karpathy's autoresearch loop to agent prompt optimization:
one file changes per round, one metric scores it, winners commit, losers revert.
The result is a compound improvement curve and a full experiment log that becomes
institutional knowledge.

---

## PM Design Decisions

### Decision 1: LLM-as-Judge for Scoring
Rather than requiring users to build a custom eval harness, the tool uses a second
LLM call (Gemini Flash) to score outputs against binary criteria. This is a deliberate
tradeoff: slightly less reliable than deterministic scoring, but it means any user can
define criteria in plain English and start a run in 15 minutes.

**Risk acknowledged:** LLM judges can be inconsistent. Mitigated by running each
test scenario 5 times per round and averaging, so single-response variance is smoothed out.

### Decision 2: Binary Criteria Only
The eval enforces yes/no questions. No 1-7 scales. This is a hard product constraint
because scales introduce subjective drift that compounds across 50 rounds and makes
scores meaningless. The UI validates this and rejects vague criteria.

### Decision 3: One Change Per Round
The loop makes exactly one edit per round so attribution is clean. If two things
change simultaneously and the score drops, you cannot know which change caused it.
This single-variable constraint is the core scientific validity of the Karpathy pattern.

### Decision 4: Git-Based Rollback
Every round uses git commit (winner) or git reset (loser). This gives you a clean
audit trail, not just a log file. You can checkout any round and inspect the exact
prompt that produced that score.

### Decision 5: The Log Is the Real Asset
The improved prompt is useful. The experiment log is more useful. It tells the next
PM exactly what was tried, what worked, and what failed. When a better model ships,
you hand it the log and say "start from experiment 20."

---

## Eval Criteria (Production-Grade)

These five criteria are designed to be actually useful at work, not just demo props.
They cover intent recognition, groundedness, actionability, concision, and honest
handling of uncertainty.

```
C1: Does the agent correctly identify the user's intent on the first response?
    PASS: User asks "how do I reset my password" → agent provides reset steps
    FAIL: User asks "how do I reset my password" → agent asks "what kind of password?"

C2: Does the response avoid making up information not in the provided context?
    PASS: "I don't have information about that specific policy. Let me connect you
          with someone who does."
    FAIL: "Our refund policy allows returns within 30 days." (when no refund policy
          was in the context)

C3: Does the response include a specific next action for the user?
    PASS: "Click Settings > Security > Reset Password, then check your email for a
          confirmation link."
    FAIL: "You should be able to find that in your account settings somewhere."

C4: Is the response under 80 words?
    PASS: 54 words
    FAIL: 150 words

C5: For impossible or ambiguous requests, does the agent say so clearly instead
    of guessing?
    PASS: "I can help with billing questions, but I'd need to transfer you to our
          technical team for server configuration."
    FAIL: "Sure, let me try to help with your server configuration." (when the agent
          has no server knowledge)
```

---

## Test Scenarios (Demo Mode)

Three scenarios designed to stress-test all five criteria:

```
Scenario 1 (Clear request, full context):
"My payment failed but I was still charged. What do I do?"

Scenario 2 (Ambiguous request, missing context):
"Can you fix my account? It's broken."

Scenario 3 (Out of scope request):
"Can you give me the CEO's direct email address?"
```

---

## The Broken System Prompt (Demo Starting Point)

This is the intentionally weak prompt the demo loop starts from. It fails C1 (vague
intent handling), C2 (no grounding instruction), C3 (no action specificity), and C5
(no honest refusal instruction):

```
You are a helpful customer support assistant. Help users with their questions.
Be polite and professional. If you don't know something, try your best to help.
Escalate to a human if needed.
```

---

## Architecture

```
autoeval-loop/
├── app.py                    # Streamlit UI (two tabs)
├── loop.py                   # Core optimization engine
├── eval_harness.py           # LLM-as-judge scoring engine
├── git_manager.py            # Git commit/reset wrapper
├── templates/                # Pre-built eval configs
│   ├── support_agent.json
│   ├── sales_assistant.json
│   ├── onboarding_bot.json
│   ├── knowledge_base_bot.json
│   └── scheduling_agent.json
├── demo/                     # Pre-loaded demo assets
│   ├── broken_prompt.txt
│   ├── eval_criteria.json
│   └── test_scenarios.json
├── results/                  # Sample run output (committed to repo)
│   ├── sample_results.log
│   └── sample_changelog.md
├── workspace/                # Runtime directory (gitignored)
│   ├── current_prompt.txt    # The one file the loop mutates
│   └── results.log           # Live experiment log
├── requirements.txt
├── .env.example
└── README.md
```

### Data Flow

```
User Input (system prompt + criteria + scenarios)
         ↓
   eval_harness.py
   ├── Generate 5 outputs per scenario via Groq (llama-3.1-8b-instant)
   ├── Score each output against each criterion via Gemini Flash
   └── Return: score %, per-criterion breakdown, raw outputs
         ↓
   loop.py
   ├── Identify lowest-scoring criterion
   ├── Ask Gemini Flash to make ONE targeted edit to current_prompt.txt
   ├── Run eval_harness again
   ├── Score improved? → git commit, update baseline
   ├── Score dropped?  → git reset HEAD~1, log as reverted
   └── Repeat until 95%+ for 3 consecutive rounds OR 50 rounds reached
         ↓
   app.py (Streamlit)
   ├── Live round-by-round progress feed
   ├── Plotly score chart (updates per round)
   ├── Per-criterion heatmap (updates per round)
   └── Final prompt diff (original vs optimized)
```

---

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Generation LLM | Groq (llama-3.1-8b-instant) | Free tier |
| Judge LLM | Gemini Flash 1.5 | Free tier |
| UI | Streamlit | Free |
| Charts | Plotly | Free |
| Git operations | GitPython | Free |
| Total per run | -- | ~$0 |

---

## File Responsibilities (The Karpathy Mapping)

| Karpathy's Repo | This Project | Rule |
|-----------------|--------------|------|
| train.py | workspace/current_prompt.txt | ONLY file the loop mutates |
| prepare.py (locked eval) | eval_harness.py | Loop can NEVER modify this |
| program.md | loop.py instruction block | Human defines strategy |
| val_bpb metric | composite score % | Single number, higher is better |
| git commit/reset | git_manager.py | Keep winners, revert losers |

---

## What to Build in Each Claude Code Session

```
Session 1: Project scaffold + git setup
Session 2: eval_harness.py (scoring engine)
Session 3: loop.py (optimization engine)
Session 4: git_manager.py (commit/reset wrapper)
Session 5: app.py Tab 1 (Try the Demo)
Session 6: app.py Tab 2 (Optimize Your Own)
Session 7: Live progress streaming in UI
Session 8: Score chart + per-criterion heatmap
Session 9: Prompt diff view + experiment log display
Session 10: Templates, demo assets, README, final polish
```

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM judge inconsistency | Run each scenario 5x, average scores |
| Loop games the eval | Eval harness is read-only, never passed to optimizer |
| Groq rate limits | Add exponential backoff, configurable delay between rounds |
| Git conflicts in workspace | Isolated workspace/ directory, fresh init per run |
| Demo too slow for recruiter | Pre-compute a sample run, offer "replay mode" |

---

## Lessons Learned Section (for README)

To be filled in after first real run. Template:

- What the loop fixed first and why (always the worst-scoring criterion)
- What it tried that made things worse (and auto-reverted)
- What the final prompt change log revealed about the original prompt's failure modes
- Why the experiment log is more valuable than the final prompt

---

## PM Framing for README Opening

> "Prompt engineering at most AI teams is informal: one person edits the system
> prompt, manually tests a few scenarios, ships it, and moves on. There is no
> scoring. No experiment log. No way to prove the new version is better.
>
> AutoEval Loop applies the Karpathy autoresearch pattern to agent prompt
> optimization. You define what 'better' means in 5 binary questions. The loop
> runs 50 experiments overnight. You wake up to a scored improvement curve,
> a per-criterion breakdown, and an experiment log that any future model can
> pick up from round 1."

---

*BLUEPRINT.md — for internal reference and portfolio documentation*
*Do not delete: this file explains design decisions that are not obvious from the code*
