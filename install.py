"""One-shot installer for the halucheck + preferences toolkit.

Idempotent. Works in a new Claude Code session OR an existing one (the
harness re-reads settings.json on the next prompt, so hooks attach without
a restart).

Steps it runs:
  1. Register the local marketplace (claude plugin marketplace add)
  2. Install the halucheck plugin (claude plugin install)
  3. Merge the UserPromptSubmit + Stop hooks into ~/.claude/settings.json
  4. Drop user_preferences.json template + the user_preferences.py hook
     into ~/.claude/
  5. Copy the four user-scope slash commands (smart_prompt, judge_response,
     check_claim, setup_prefs, prefs) into ~/.claude/commands/

Usage:
    python install.py             # install
    python install.py --uninstall # tear down

Optional flags:
    --no-plugin    skip plugin install (just hooks + commands)
    --no-hooks     skip settings.json patching
    --proxy-only   only verify the proxy is running, do nothing else
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent           # ...\gpt2_vg
HOME = Path.home()
CLAUDE_DIR = HOME / ".claude"

PLUGIN_CACHE_GLOB = "plugins/cache/halucheck-local/halucheck/*/hooks"

# Files we own
USER_PREF_TEMPLATE = REPO / "claude_setup" / "user_preferences.json"
USER_PREF_HOOK = REPO / "claude_setup" / "hooks" / "user_preferences.py"
COMMAND_FILES = [
    REPO / "claude_setup" / "commands" / name for name in
    ("smart_prompt.md", "judge_response.md", "check_claim.md",
     "setup_prefs.md", "prefs.md", "install_halucheck.md")
]

# Sentinel keys we add to settings.json so uninstall can find what we own
HOOK_TAG = "__installed_by_halucheck_install_py__"


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess with CLAUDECODE unset so the `claude` CLI doesn't
    refuse to start when invoked from inside a Claude Code session.

    On Windows `claude` is a `.cmd` shim which subprocess can't resolve
    with shell=False (WinError 2); route through the shell there with a
    properly-quoted command line.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    if os.name == "nt":
        return subprocess.run(subprocess.list2cmdline(args), env=env,
                              capture_output=True, text=True, shell=True, **kw)
    return subprocess.run(args, env=env, capture_output=True, text=True, **kw)


def _latest_plugin_hook_dir() -> Path | None:
    matches = sorted(CLAUDE_DIR.glob(PLUGIN_CACHE_GLOB))
    return matches[-1] if matches else None


# ---------- step 1: marketplace + plugin ----------

def install_plugin() -> None:
    # Marketplace (idempotent: re-adding the same source is a no-op error)
    r = _run(["claude", "plugin", "marketplace", "add", str(REPO) + "/"])
    if r.returncode != 0 and "already" not in (r.stdout + r.stderr).lower():
        print(f"  marketplace add stderr: {r.stderr.strip()}", file=sys.stderr)
    print(f"  [OK] marketplace registered")

    r = _run(["claude", "plugin", "install", "halucheck@halucheck-local"])
    out = (r.stdout + r.stderr).strip()
    if "Successfully installed" in out or "already" in out.lower():
        print(f"  [OK] plugin installed")
    else:
        print(f"  [WARN] plugin install: {out}")


def patch_plugin_ckpt_root() -> None:
    """Rewrite HALUCHECK_CKPT_ROOT in the installed plugin.json to point at
    the source repo location (wherever install.py was run from). Without
    this, the plugin's MCP server can't find ckpt_judge_*/ — they live in
    the project root, not bundled in the plugin."""
    cached = sorted(CLAUDE_DIR.glob("plugins/cache/halucheck-local/halucheck/*/.claude-plugin/plugin.json"))
    if not cached:
        # New install pattern may live without .claude-plugin/ — check legacy too
        cached = sorted(CLAUDE_DIR.glob("plugins/cache/halucheck-local/halucheck/*/plugin.json"))
    for pj in cached:
        try:
            cfg = json.loads(pj.read_text(encoding="utf-8-sig"))
            env = cfg.get("mcpServers", {}).get("halucheck", {}).get("env", {})
            if env.get("HALUCHECK_CKPT_ROOT") != str(REPO):
                env["HALUCHECK_CKPT_ROOT"] = str(REPO)
                pj.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                print(f"  [OK] patched HALUCHECK_CKPT_ROOT in {pj}")
        except Exception as e:
            print(f"  [WARN] could not patch {pj}: {e}")


def uninstall_plugin() -> None:
    _run(["claude", "plugin", "uninstall", "halucheck"])
    _run(["claude", "plugin", "marketplace", "remove", "halucheck-local"])
    print(f"  [OK] plugin + marketplace removed")


# ---------- step 2: settings.json hooks ----------

def _read_settings() -> dict:
    p = CLAUDE_DIR / "settings.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _write_settings(cfg: dict) -> None:
    p = CLAUDE_DIR / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def install_hooks() -> None:
    hook_dir = _latest_plugin_hook_dir()
    user_pref_hook_dst = CLAUDE_DIR / "hooks" / "user_preferences.py"

    user_pref_cmd = (f"python {user_pref_hook_dst.as_posix()}")
    preflight_cmd = (f"python {(hook_dir / 'user_prompt_check.py').as_posix()}"
                     if hook_dir else None)
    stop_cmd = (f"python {(hook_dir / 'stop_judge.py').as_posix()}"
                if hook_dir else None)

    cfg = _read_settings()
    cfg.setdefault("hooks", {})

    # UserPromptSubmit: [user_preferences, halucheck_preflight]
    ups_inner_hooks = [
        {"type": "command", "command": user_pref_cmd, "timeout": 5,
         HOOK_TAG: "user_preferences"},
    ]
    if preflight_cmd:
        ups_inner_hooks.append({
            "type": "command", "command": preflight_cmd, "timeout": 10,
            HOOK_TAG: "halucheck_preflight",
        })
    cfg["hooks"]["UserPromptSubmit"] = [{"matcher": "", "hooks": ups_inner_hooks}]

    # Stop: halucheck_stop_judge
    if stop_cmd:
        cfg["hooks"]["Stop"] = [{
            "matcher": "",
            "hooks": [{
                "type": "command", "command": stop_cmd, "timeout": 10,
                HOOK_TAG: "halucheck_stop_judge",
            }],
        }]

    _write_settings(cfg)
    print(f"  [OK] hooks registered in {CLAUDE_DIR / 'settings.json'}")


def uninstall_hooks() -> None:
    cfg = _read_settings()
    hooks = cfg.get("hooks", {})
    for event, entries in list(hooks.items()):
        new_entries = []
        for entry in entries:
            inner = entry.get("hooks", [])
            kept = [h for h in inner if not h.get(HOOK_TAG)]
            if kept:
                entry = dict(entry, hooks=kept)
                new_entries.append(entry)
        if new_entries:
            hooks[event] = new_entries
        else:
            hooks.pop(event, None)
    _write_settings(cfg)
    print(f"  [OK] hooks removed from settings.json")


# ---------- step 3: user preferences + commands ----------

def _cwd_project_commands_dir() -> Path | None:
    """If cwd looks like a Claude Code project root that ISN'T the source repo,
    return its .claude/commands/ so we can drop slash commands there too.
    CCD discovers slash commands per-project, not from ~/.claude/commands/."""
    cwd = Path.cwd().resolve()
    if cwd == REPO:
        return None  # source repo handled separately
    return cwd / ".claude" / "commands"


def install_user_files() -> None:
    (CLAUDE_DIR / "hooks").mkdir(parents=True, exist_ok=True)
    (CLAUDE_DIR / "commands").mkdir(parents=True, exist_ok=True)
    project_cmds = _cwd_project_commands_dir()
    if project_cmds:
        project_cmds.mkdir(parents=True, exist_ok=True)

    # user_preferences.json template (preserve existing if user has tuned it)
    upj = CLAUDE_DIR / "user_preferences.json"
    if not upj.exists() and USER_PREF_TEMPLATE.exists():
        shutil.copy2(USER_PREF_TEMPLATE, upj)
        print(f"  [OK] wrote default {upj}")
    elif upj.exists():
        print(f"  [OK] kept existing {upj} (no overwrite)")

    # hook script
    if USER_PREF_HOOK.exists():
        shutil.copy2(USER_PREF_HOOK, CLAUDE_DIR / "hooks" / "user_preferences.py")
        print(f"  [OK] user_preferences.py hook in place")

    # slash commands — drop into ~/.claude/commands/ AND <cwd>/.claude/commands/
    # if cwd is a different project (CCD discovers per-project).
    installed = 0
    for src in COMMAND_FILES:
        if src.exists():
            shutil.copy2(src, CLAUDE_DIR / "commands" / src.name)
            if project_cmds:
                shutil.copy2(src, project_cmds / src.name)
            installed += 1
    if project_cmds:
        print(f"  [OK] {installed} slash commands installed "
              f"(user scope + {project_cmds})")
    else:
        print(f"  [OK] {installed} slash commands installed (user scope)")


def uninstall_user_files() -> None:
    for src in COMMAND_FILES:
        dst = CLAUDE_DIR / "commands" / src.name
        if dst.exists():
            dst.unlink()
    hook = CLAUDE_DIR / "hooks" / "user_preferences.py"
    if hook.exists():
        hook.unlink()
    print(f"  [OK] slash commands + user_preferences.py removed "
          "(user_preferences.json kept in case you want to restore)")


# ---------- proxy check ----------

def check_proxy() -> None:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as r:
            j = json.loads(r.read().decode())
        print(f"  [OK] proxy up at localhost:8000 (rephrase={j.get('rephrase')}, "
              f"judge={j.get('judge')})")
    except Exception as e:
        print(f"  [WARN] proxy NOT reachable on :8000. Start it with:")
        print(f"      cd {REPO} && HALUCHECK_JUDGE_LORA=ckpt_judge_enriched "
              f"HALUCHECK_JUDGE_LORA_V2=ckpt_judge_enriched_v2 "
              f"python halucheck_proxy.py")
        print(f"    (hooks will silently no-op without the proxy; not fatal.)")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--no-plugin", action="store_true")
    ap.add_argument("--no-hooks", action="store_true")
    ap.add_argument("--proxy-only", action="store_true")
    args = ap.parse_args()

    if args.proxy_only:
        check_proxy()
        return 0

    if args.uninstall:
        print("Uninstalling halucheck toolkit…")
        uninstall_hooks()
        uninstall_user_files()
        uninstall_plugin()
        print("Done. Restart your Claude Code session to clear runtime state.")
        return 0

    print(f"Installing halucheck toolkit from {REPO}")
    print(f"Target: {CLAUDE_DIR}")

    if not args.no_plugin:
        print("[1/4] plugin")
        install_plugin()
        patch_plugin_ckpt_root()

    print("[2/4] user files (preferences config, hook, slash commands)")
    install_user_files()

    if not args.no_hooks:
        print("[3/4] hooks in settings.json")
        install_hooks()

    print("[4/4] proxy check")
    check_proxy()

    print()
    print("Done. Hooks attach on your next prompt. Run /setup_prefs to configure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
