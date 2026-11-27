"""Require the root's single fresh owner, current holder release and exact CPU witness."""
from datetime import datetime
import json
import os
import re
import subprocess

from cp3_contract import ARROW_RESULT, ARROW_ROOT, LABEL, REQUEST, need


PLACEMENT = dict(parent="1103606", node="f13r1n14", dcu=8, cpus=64, mem_mib=224000, port=29991)
CPU_ROW = ["1103606.4726", "internvl-gqa56-arrow-e1-c6d2ed29e945", "COMPLETED", "0:0",
           "29", "f13r1n14", "cpu=8,mem=24G,node=1", "2026-11-27T18:28:33", "2026-11-27T18:29:02"]
COLS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
SHA = re.compile(r"[a-f0-9]{64}")


def exact_accounting(step, run=subprocess.run):
    result = run(["sacct", "-nP", "-j", step, "-o", COLS], stdin=subprocess.DEVNULL,
                 capture_output=True, timeout=8, check=False,
                 env={**os.environ, "TZ": "UTC", "LC_ALL": "C", "SLURM_TIME_FORMAT": "standard"})
    need(result.returncode == 0 and not result.stderr and len(result.stdout) <= 65536,
         "EXACT_QUIET_CP3_ACCOUNTING")
    rows = [line.split("|") for line in result.stdout.decode().splitlines()
            if line.split("|")[0] == step]
    need(len(rows) == 1 and len(rows[0]) == 9, "EXACT_CP3_ACCOUNTING_ROW")
    return rows[0]


def verify_cpu_accounting(run=subprocess.run):
    row = exact_accounting(CPU_ROW[0], run)
    need(row == CPU_ROW, "EXACT_4726_ZERO_GRES_TERMINAL")
    return row


def validate_release(release, issued, expected_step):
    row = release["accounting"]
    need(release["request"] == REQUEST and release["parent"] == PLACEMENT["parent"] and
         release["node"] == PLACEMENT["node"] and
         release["holder_step"] == expected_step and release["numbered_steps"] == release["queued_steps"] == [] and
         release["worker_returned"] is True and
         all(SHA.fullmatch(release[key]) for key in ("expected_file_SHA256", "worker_file_SHA256")),
         "EXACT_FRESH_OWNED_HOLDER_RELEASE")
    need(len(row) == 9 and row[0] == expected_step and row[2:4] == ["COMPLETED", "0:0"] and
         re.fullmatch(r"training80-dcu-standby-[0-9a-f]+-f13r1n14", row[1]) and row[4].isdigit() and
         row[5:7] == ["f13r1n14", "cpu=8,gres/dcu=8,mem=16G,node=1"], "EXACT_REAL_HOLDER_TERMINAL")
    start, end = (datetime.fromisoformat(value + "+00:00").timestamp() for value in row[7:])
    need(start <= end <= release["observed_epoch"] <= issued and issued-release["observed_epoch"] <= 45,
         "EXACT_CP3_FRESH_RELEASE_CHRONOLOGY")
    return row


def load(here, authority_sha, publication, claim_sha, manifest_sha, pin, now):
    need(SHA.fullmatch(authority_sha or ""), "NEW_CP3_ROOT_AUTHORITY_REQUIRED")
    owner = json.loads(pin(here/"owner-authority.json", expected=authority_sha, cap=8192))
    need((owner["request"], owner["label"], owner["publication"], owner["claim_SHA256"],
          owner["manifest_SHA256"], owner["placement"], owner["zero_gres_step"],
          owner["zero_gres_result_SHA256"]) ==
         (REQUEST, LABEL, publication, claim_sha, manifest_sha, PLACEMENT, CPU_ROW[0], ARROW_RESULT)
         and re.fullmatch(r"[a-f0-9]{40}", publication)
         and all(SHA.fullmatch(value) for value in (claim_sha, manifest_sha, owner["release_evidence_SHA256"]))
         and owner["GPU_GO"] is True and owner["replay_allowed"] is False
         and owner["issued_epoch"] <= now < owner["expires_epoch"] == owner["issued_epoch"]+120,
         "EXACT_FRESH_CP3_OWNER")
    holder = owner["latest_holder_step"]
    need(re.fullmatch(r"1103606\.[0-9]+", holder) and int(holder.split(".")[1]) >= 4708,
         "EXACT_CURRENT_HOLDER_SELECTION")
    release = json.loads(pin(here/"holder-release.json", expected=owner["release_evidence_SHA256"], cap=8192))
    observed = validate_release(release, owner["issued_epoch"], holder)
    pin(ARROW_ROOT/"result.json", expected=ARROW_RESULT, extent=6740)
    return dict(owner, holder_terminal_row=observed)
