"""Add exact-node reservation recognition to the frozen 05:44 ordinary coordinator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import arbiter


ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/run_training80_0544_ordinary_refill_once.py"
ENTRY_SHA = "2fb4649087e1e2162f706fe61b095cc434c120758bff4e60f843ebc973de7a0b"
GROUP = 6


def original(journal=None):
    data = ENTRY.read_bytes()
    arbiter.require(arbiter.signature(data) == ENTRY_SHA, "ORIGINAL_0544_ENTRY")
    spec = importlib.util.spec_from_file_location("pack16_original_0544", ENTRY)
    wrapper = importlib.util.module_from_spec(spec)
    exec(compile(data, str(ENTRY), "exec"), vars(wrapper))
    coordinator = {"__name__": "pack16_original_coordinator", "__file__": str(ENTRY)}
    exec(wrapper.scope_source(), coordinator)
    groups = coordinator["groups"]()
    selected = [row for row in groups[GROUP - 1] if row["node"] == arbiter.NODE]
    arbiter.require(len(selected) == 1 and selected[0]["parent"] == arbiter.PARENT
                    and selected[0]["previous"] == arbiter.PREDECESSOR
                    and sum(len(group) for group in groups) == 70
                    and coordinator["CALENDAR_SHA"] == arbiter.CALENDAR_SHA
                    and any(row["node"] == "f13r1n12" for row in groups[10]), "ORIGINAL_0544_SCOPE")
    ledger = arbiter.Ledger(coordinator["JOURNAL"] if journal is None else journal)
    prior_reserve = coordinator["reserve"]

    def reserve(node, step, intent, base):
        # Reuse the original durable ticket operation inside the same exact-node lock.
        return ledger.ordinary(node, step, lambda: prior_reserve(node, step, intent, base))

    coordinator["reserve"] = reserve
    if journal is not None:
        coordinator["JOURNAL"] = journal
    # All original calculations ran with the upstream source path. Its child now
    # re-enters this same scoped adapter so the frozen group logic stays in use.
    coordinator["__file__"] = str(Path(__file__).resolve())
    return coordinator, ledger, GROUP


def refill_argv(group):
    arbiter.require(group == GROUP, "ONLY_ORIGINAL_GROUP_6_RELEASE")
    return [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--resume-pack16"]


def resume_inputs(coordinator, ledger):
    with ledger.guarded():
        accepted, accepted_sha = ledger.get("accepted")
        release, _ = ledger.get("release")
        arbiter.require(accepted and release and release["accepted_SHA256"] == accepted_sha
                        and release["group"] == GROUP and release["state"] in (
                            "RELEASED_POSITIVE_UNSENT", "RELEASED_AFTER_TERMINAL_AND_POST"), "PACK16_REFILL_NOT_RELEASED")
    return [row for row in coordinator["groups"]()[GROUP - 1]
            if row["node"] == arbiter.NODE and row["previous"] == arbiter.PREDECESSOR]


def main():
    coordinator, ledger, group = original()
    if sys.argv[1:] == ["--resume-pack16"]:
        try:
            ready = resume_inputs(coordinator, ledger)
            base, publisher = coordinator["modules"]()
            result = coordinator["refill"](group, ready, base, publisher)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            print(type(error).__name__, file=sys.stderr)
            return 1
        print(arbiter.encode(result).decode(), end="")
        return 0
    if sys.argv[1:] == ["--check"]:
        sys.argv.pop()
    try:
        coordinator["main"]()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
