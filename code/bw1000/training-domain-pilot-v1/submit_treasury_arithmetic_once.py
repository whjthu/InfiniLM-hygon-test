"""Stage the published Treasury arithmetic follow-up and submit it once."""

import argparse
import base64
from hashlib import sha256
import json
import re
import runpy
import subprocess
import uuid

from pathlib import Path

from render_treasury_arithmetic_retry import LABEL, controller, generation
from submit_tum_native_once import G, pin


SSH_HELPER_SHA256 = "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"
DOMAIN = "code/bw1000/training-domain-pilot-v1/"
TRAIN_NATIVE = "code/bw1000/training-qwen-native25-hip-v3/"
ORIGINAL = "training-qwen-treasury-dev-pair-hip-v2"
HOST = r"""
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def stage():
    os.umask(0o077)
    epoch = time.time()
    results = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
    root = results / CLAIM['label']
    output = subprocess.check_output(
        ['sacct', '-n', '-P', '-j', CLAIM['parent'], '-o', 'JobIDRaw,JobName%100,State,ExitCode'],
        timeout=12, text=True)
    rows = [line.split('|') for line in output.splitlines()]
    labels = {CLAIM['label'] + '-' + phase for phase in ('pre', 'gpu', 'post')}
    if any(len(row) == 4 and row[1] in labels for row in rows):
        raise ValueError('existing numbered accounting')
    terms = {'1087656.298': 'training-qwen-treasury-dev-pair-hip-v2-gpu',
             '1087656.299': 'training-qwen-treasury-dev-pair-hip-v2-post'}
    for handle, label in terms.items():
        if rows.count([handle, label, 'COMPLETED', '0:0']) != 1:
            raise ValueError('unconfirmed original terminal ownership')
    assets = (results / CLAIM['original_label'] / 'src' / 'MODEL_ASSETS.json').read_bytes()
    sources = {name: base64.b64decode(raw, validate=True) for name, raw in SOURCES.items()}
    sources['MODEL_ASSETS.json'] = assets
    if {name: sha256(raw).hexdigest() for name, raw in sources.items()} != CLAIM['source_SHA256']:
        raise ValueError('staged source pin differs')
    claim = (json.dumps(CLAIM, sort_keys=True, separators=(',', ':')) + '\n').encode()
    root.mkdir()
    intent = dict(request=REQUEST, publication=PUBLICATION, claim_SHA256=sha256(claim).hexdigest(),
                  epoch=epoch, deadline=epoch + CLAIM['deadline_seconds'])
    with (root / 'intent.json').open('x') as f:
        json.dump(intent, f)
    print(json.dumps(dict(stage='intent', request=REQUEST, label=CLAIM['label'])), flush=True)
    with (root / 'claim.json').open('xb') as f:
        f.write(claim)
    (root / 'src').mkdir()
    for name, raw in sources.items():
        with (root / 'src' / name).open('xb') as f:
            f.write(raw)
    return os.spawnv(os.P_WAIT, sys.executable, [
        sys.executable, '-I', '-S', '-B', '-c',
        'import runpy,sys; p=sys.argv[1]; sys.path.insert(0,p.rsplit("/",1)[0]); '
        'sys.argv=[p,"controller"]; runpy.run_path(p,run_name="__main__")',
        str(root / 'src' / 'dev_pair_controller.py')])


try:
    rc = stage()
except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
    print(json.dumps(dict(stage='staging_failed', error_class=type(exc).__name__)), flush=True)
    rc = 1
raise SystemExit(rc)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.publication):
        raise ValueError("expected an exact published commit")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + G.as_posix(), "-C", str(G)]

    def from_git(name):
        return subprocess.check_output([*git, "show", args.publication + ":" + name], timeout=10)

    head = subprocess.check_output([*git, "ls-remote", "--exit-code", "origin",
                                   "refs/heads/bw1000-experiments"], timeout=15).decode().split()[0]
    if head != args.publication:
        raise ValueError("expected the exact published claim")
    claim = json.loads(from_git("coordination/tasks/" + LABEL + ".json"))
    if (claim["label"], claim["original_label"], claim["parent"], claim["node"], claim["deadline_seconds"],
        claim["max_gpu_invocations"], claim["planned_rows_per_arm"], claim["decode"]["max_new_tokens"]) != (
            LABEL, ORIGINAL, "1087656", "a01r4n14", 3000, 1, 36, 128):
        raise ValueError("candidate scope differs")
    files = {name: from_git(DOMAIN + name) for name in ("check_qwen_holdout_native.py", "check_qwen_native_inputs.py")}
    files.update({name: from_git(TRAIN_NATIVE + name) for name in ("snapshot.py", "slurm_accounting.py")})
    files["dev_pair_controller.py"] = controller(from_git(DOMAIN + "dev_pair_controller.py"))
    files["dev_pair_generation.py"] = generation(from_git(DOMAIN + "dev_pair_generation.py"))
    pins = claim["source_SHA256"]
    if {name: sha256(raw).hexdigest() for name, raw in files.items()} != {
            name: value for name, value in pins.items() if name != "MODEL_ASSETS.json"}:
        raise ValueError("published source pins differ")
    pin(args.ssh_helper.read_bytes(), SSH_HELPER_SHA256)
    ssh = runpy.run_path(str(args.ssh_helper))["ssh"]
    request = uuid.uuid4().hex
    sources = {name: base64.b64encode(raw).decode("ascii") for name, raw in files.items()}
    block = f"CLAIM={claim!r}\nREQUEST={request!r}\nPUBLICATION={args.publication!r}\nSOURCES={sources!r}\n"
    body = (block + HOST).encode("utf8")
    out = G / "local-only" / LABEL
    out.mkdir()
    (out / "request.json").write_text(json.dumps(dict(request=request, publication=args.publication,
                                                     body_SHA256=sha256(body).hexdigest())), encoding="ascii")
    print(json.dumps(dict(stage="request", request=request)), flush=True)
    with (out / "stdout").open("xb") as stdout, (out / "stderr").open("xb") as stderr:
        try:
            run = subprocess.run([*ssh, "/usr/local/python3.12/bin/python3 -I -S -B -"], input=body,
                                 stdout=stdout, stderr=stderr, timeout=3030)
        except subprocess.TimeoutExpired:
            run = None
    print(json.dumps(dict(request=request, ssh_rc=run.returncode if run else None,
                          stdout_SHA256=sha256((out / "stdout").read_bytes()).hexdigest(),
                          stderr_SHA256=sha256((out / "stderr").read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
