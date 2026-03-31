from __future__ import annotations

import difflib
import pathlib
from datetime import datetime, timezone

import git


PROMPT_FILE = "current_prompt.txt"
LOG_FILE = "results.log"


class WorkspaceNotInitializedError(Exception):
    """Raised when workspace git repo does not exist. Call init_workspace() first."""


class PromptGitManager:
    def __init__(self, workspace_path: str = "workspace/") -> None:
        self.workspace = pathlib.Path(workspace_path)
        self.prompt_file = self.workspace / PROMPT_FILE
        self.log_file = self.workspace / LOG_FILE

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def init_workspace(self, initial_prompt: str) -> None:
        """Write initial prompt, init git repo if needed, make baseline commit."""
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Init repo only if .git does not already exist
        git_dir = self.workspace / ".git"
        if git_dir.exists():
            repo = git.Repo(str(self.workspace))
        else:
            repo = git.Repo.init(str(self.workspace))

        self.prompt_file.write_text(initial_prompt, encoding="utf-8")

        repo.index.add([PROMPT_FILE])
        repo.index.commit("baseline: round 0")

    # ------------------------------------------------------------------
    # Prompt read / write
    # ------------------------------------------------------------------

    def read_current_prompt(self) -> str:
        self._require_repo()
        return self.prompt_file.read_text(encoding="utf-8")

    def write_candidate_prompt(self, prompt: str) -> None:
        self._require_repo()
        self.prompt_file.write_text(prompt, encoding="utf-8")

    # ------------------------------------------------------------------
    # Commit / revert
    # ------------------------------------------------------------------

    def commit_round(
        self, round_num: int, score: float, change_description: str
    ) -> None:
        repo = self._require_repo()
        repo.index.add([PROMPT_FILE])
        msg = f"round {round_num}: score {score:.1f}% - {change_description} KEPT"
        repo.index.commit(msg)

    def revert_round(
        self, round_num: int, score: float, change_description: str
    ) -> None:
        repo = self._require_repo()
        repo.git.reset("--hard", "HEAD~1")
        entry = f"ROUND {round_num}: REVERTED - {change_description}"
        self.write_log_entry(entry)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_commit_history(self) -> list[dict]:
        repo = self._require_repo()
        history = []
        for commit in repo.iter_commits():
            msg = commit.message.strip()
            round_num = None
            score = None
            if msg.startswith("round "):
                parts = msg.split(":")
                try:
                    round_num = int(parts[0].split()[1])
                    score_part = parts[1].strip().split("%")[0].split()[-1]
                    score = float(score_part)
                except (IndexError, ValueError):
                    pass
            elif msg.startswith("baseline"):
                round_num = 0
                score = None

            history.append({
                "round": round_num,
                "score": score,
                "message": msg,
                "timestamp": datetime.fromtimestamp(
                    commit.committed_date, tz=timezone.utc
                ).isoformat(),
            })
        return history

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def write_log_entry(self, entry: str) -> None:
        self._require_repo()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] {entry}\n"
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line)

    def get_full_log(self) -> str:
        self._require_repo()
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def get_prompt_diff(self, original_prompt: str) -> str:
        self._require_repo()
        current = self.read_current_prompt()
        diff_lines = difflib.unified_diff(
            original_prompt.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="original_prompt.txt",
            tofile="current_prompt.txt",
        )
        return "".join(diff_lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_repo(self) -> git.Repo:
        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            raise WorkspaceNotInitializedError(
                f"No git repo found at '{self.workspace}'. "
                "Call init_workspace(initial_prompt) first."
            )
        return git.Repo(str(self.workspace))


# ---------------------------------------------------------------------------
# __main__ — exercise every method in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil

    TEST_WORKSPACE = "workspace_test/"
    ORIGINAL_PROMPT = (
        "You are a helpful assistant. Answer all questions as best you can."
    )
    IMPROVED_PROMPT_V1 = (
        "You are a helpful assistant. Always identify the user's intent before answering. "
        "Provide a specific next action in every response."
    )
    IMPROVED_PROMPT_V2 = (
        "You are a helpful assistant. Always identify the user's intent before answering. "
        "Provide a specific next action. Keep responses under 80 words."
    )
    BAD_PROMPT = (
        "You are an assistant. Help."
    )

    # Clean slate
    shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)

    mgr = PromptGitManager(workspace_path=TEST_WORKSPACE)

    # --- Test: error before init ---
    print("1. Accessing uninitialised workspace should raise an error...")
    try:
        mgr.read_current_prompt()
        print("   FAIL: no error raised")
    except Exception as exc:
        print(f"   OK: {exc}")

    # --- init_workspace ---
    print("\n2. init_workspace()...")
    mgr.init_workspace(ORIGINAL_PROMPT)
    print(f"   Wrote: {mgr.prompt_file}")

    # --- read_current_prompt ---
    print("\n3. read_current_prompt()...")
    txt = mgr.read_current_prompt()
    assert txt == ORIGINAL_PROMPT, "prompt mismatch"
    print(f"   OK: '{txt[:60]}…'")

    # --- write_candidate + commit (round 1, winner) ---
    print("\n4. write_candidate_prompt() + commit_round(1, 72.0, …)...")
    mgr.write_candidate_prompt(IMPROVED_PROMPT_V1)
    mgr.commit_round(1, 72.0, "added intent identification and next-action instruction")
    mgr.write_log_entry("ROUND 1: KEPT — score 72.0%")
    print("   OK")

    # --- write_candidate + commit + revert (round 2, loser) ---
    # Workflow: always commit the candidate first, then revert if score dropped.
    # revert_round does HEAD~1 --hard, so the commit must exist before calling it.
    print("\n5. write_candidate_prompt() + commit_round(2) + revert_round(2, 65.0, …)...")
    mgr.write_candidate_prompt(BAD_PROMPT)
    mgr.commit_round(2, 65.0, "stripped prompt too aggressively")
    mgr.revert_round(2, 65.0, "stripped prompt too aggressively — score dropped")
    current_after_revert = mgr.read_current_prompt()
    assert current_after_revert == IMPROVED_PROMPT_V1, (
        f"Revert did not restore v1.\nGot: {current_after_revert!r}"
    )
    print("   OK: prompt correctly restored to v1 after revert")

    # --- write_candidate + commit (round 3, winner) ---
    print("\n6. commit_round(3, 85.0, …)...")
    mgr.write_candidate_prompt(IMPROVED_PROMPT_V2)
    mgr.commit_round(3, 85.0, "added 80-word concision rule")
    mgr.write_log_entry("ROUND 3: KEPT — score 85.0%")
    print("   OK")

    # --- get_commit_history ---
    print("\n7. get_commit_history()...")
    history = mgr.get_commit_history()
    for entry in history:
        print(f"   round={entry['round']:>2}  score={str(entry['score']):>6}  {entry['message'][:60]}")

    # --- get_prompt_diff ---
    print("\n8. get_prompt_diff(original_prompt)...")
    diff = mgr.get_prompt_diff(ORIGINAL_PROMPT)
    if diff:
        print(diff)
    else:
        print("   (no diff — prompts are identical)")

    # --- get_full_log ---
    print("\n9. get_full_log()...")
    log = mgr.get_full_log()
    print(log if log else "   (log is empty)")

    # Cleanup
    shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)
    print("10. Test workspace cleaned up. All tests passed.")
