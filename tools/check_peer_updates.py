"""Read-only Git polling; only ignored local state is written. No peer code runs."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def relevant(path, peer):
    return path in {"AGENTS.md", "README.md"} or path.startswith((
        "coordination/", "handoff/", f"status/{peer}/", f"results/{peer}/",
        f"analysis/{peer}/", f"code/{peer}/", f"manifests/{peer}/"))


def run(repo, *args):
    command = ["git", "-c", "safe.directory=" + repo.as_posix(), "-c",
               "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes",
               *args]
    proc = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    if proc.returncode:
        raise RuntimeError(f"Git {args[0]} failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return proc.stdout.strip()


def save(path, state):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(repo, side, ack=False):
    peer = "bw1100" if side == "bw1000" else "bw1000"
    directory = repo / "local-only"
    directory.mkdir(exist_ok=True)
    path = directory / f"peer-check-{side}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema_version": 1, "self": side, "peer": peer, "consumed_heads": {}}
    if ack:
        if not state.get("last_successful_fetch_utc") or state.get("last_error"):
            raise RuntimeError("No successful, error-free fetched snapshot to acknowledge")
        state["consumed_heads"] = dict(state["observed_heads"])
        state["last_consumed_utc"] = utc()
        state["pending"] = []
        save(path, state)
        return {"acknowledged": state["consumed_heads"], "remote_write": False}
    state["last_attempt_utc"] = utc()
    try:
        remote = run(repo, "remote", "get-url", "origin")
        if remote not in {"git@github.com:whjthu/InfiniLM-hygon-test.git",
                          "https://github.com/whjthu/InfiniLM-hygon-test.git",
                          "ssh://git@github.com/whjthu/InfiniLM-hygon-test.git"}:
            raise RuntimeError("Origin is not the explicitly authorized collaboration repository")
        run(repo, "fetch", "--quiet", "--no-recurse-submodules", "origin")
        refs = run(repo, "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin/")
        available = dict(line.split(" ", 1) for line in refs.splitlines() if line)
        main = "origin/main"
        branch = f"origin/{peer}-experiments"
        if main not in available:
            raise RuntimeError("origin/main missing; do not replace prior consumption cursor")
        observed = {name: available[name] for name in (main, branch) if name in available}
        pending = []
        for name, head in observed.items():
            consumed = state["consumed_heads"].get(name)
            if consumed == head:
                continue
            if consumed:
                listing = run(repo, "diff", "--name-only", consumed, head, "--")
            else:
                listing = run(repo, "ls-tree", "-r", "--name-only", head)
            paths = [p for p in listing.splitlines() if name == branch or relevant(p, peer)]
            # A new main commit with only the local side's results is not a peer message.
            if paths:
                pending.append({"ref": name, "from_commit": consumed, "to_commit": head,
                                "changed_paths": paths})
        peer_present = bool(run(repo, "ls-tree", "-r", "--name-only", observed[main], "--", f"status/{peer}/"))
        state.update(last_successful_fetch_utc=utc(), observed_heads=observed,
                     pending=pending, peer_status_present=peer_present,
                     peer_branch_present=branch in available, last_error=None)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        state["last_error"] = {"at_utc": utc(), "message": str(exc)}
        save(path, state)
        raise
    save(path, state)
    return {"checked_at_utc": state["last_successful_fetch_utc"], "peer": peer,
            "peer_status_present": peer_present, "peer_branch_present": branch in available,
            "pending": pending, "remote_write": False, "local_state": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self", dest="side", choices=("bw1000", "bw1100"), required=True)
    parser.add_argument("--ack", action="store_true", help="Consume last read snapshot without fetching")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    try:
        print(json.dumps(check(repo, args.side, args.ack), ensure_ascii=False, indent=2))
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": str(exc), "remote_write": False}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
