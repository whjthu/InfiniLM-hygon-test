"""Check ordinary SSH Git before device admission, then run the frozen renewal."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from time import monotonic, time
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "local-only/run-training80-2035-ordinary-renewal-once.py"
FROZEN_SHA256 = "62156f03554141703c55746165e4be8f7e41f4abb7b192d079bc764eb963e720"
REFILL = REPO.parent / "run-dcu-standby-cycle-20261123.ps1"
REFILL_SHA256 = "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73"
JOURNAL = REPO / "local-only/training80-1738-ordinary-renewal-dispatch"
ORIGINAL_TIMEOUT_PROOF = REPO / "tools/confirm_standby_prepublication_timeout.py"
ORIGINAL_TIMEOUT_PROOF_SHA256 = "f0b496d243ec2c113935ea7e9f633b3b4af7a3ba5c224c216d00a3852906bc42"
ORIGIN = "git@github.com:whjthu/InfiniLM-hygon-test.git"
BRANCH = "refs/heads/bw1000-experiments"
MEMBERS = tuple(f"code/bw1000/fleet-dcu-standby-v2/{name}" for name in (
    "CLAIM.json", "worker.py", "controller.py", "dispatch.py", "status.py"))
BRIDGE = tuple(f"code/bw1000/fleet-dcu-standby-compute-bridge-v1/{name}" for name in ("bridge.py", "CLAIM.json"))
ALL = MEMBERS + BRIDGE
ADMISSION_WINDOW_SECONDS = 65


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def git(*args, timeout=6):
    return subprocess.check_output(["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix(), *args],
                                   cwd=REPO, timeout=timeout, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def local_head():
    head = git("rev-parse", "HEAD").decode().strip()
    need(re.fullmatch(r"[a-f0-9]{40}", head), "local publication tip is invalid")
    need(git("remote", "get-url", "origin").decode().strip() == ORIGIN, "publication origin changed")
    return head


def checksums(paths):
    result = {}
    for name in paths:
        source = REPO / name
        need(not source.is_symlink() and source.resolve(strict=True).is_relative_to(REPO), "publication file is a link")
        result[name] = digest(source.read_bytes())
    return result


def precheck():
    head = local_head()
    started = monotonic()
    remote = git("ls-remote", "--exit-code", "origin", BRANCH, timeout=58).decode().split()
    observed = monotonic()
    need(remote == [head, BRANCH], "remote tip differs from current local publication")
    blobs = {name: digest(git("cat-file", "blob", f"{head}:{name}")) for name in ALL}
    need(blobs == checksums(ALL) and local_head() == head, "one or more published standby artifacts differ")
    need(monotonic() - observed <= 10, "published file verification exceeded its pre-admission window")
    return dict(head=head, blobs=blobs, remote_elapsed=observed-started, deadline=observed+ADMISSION_WINDOW_SECONDS)


def verify_bound(state, repo, publication, paths):
    need(repo.resolve() == REPO and publication == state["head"] and tuple(paths) in (MEMBERS, BRIDGE, ALL),
         "standby attempted an unbound source scope")
    need(monotonic() < state["deadline"] and local_head() == state["head"], "prechecked publication expired or changed")
    need(checksums(ALL) == state["blobs"] and monotonic() < state["deadline"], "prechecked standby bytes changed or expired")


def render():
    raw = REFILL.read_bytes()
    need(digest(raw) == REFILL_SHA256, "frozen device/dispatch runner changed")
    source = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n@'\n", 1)[1].split("\n'@ | & ", 1)[0]
    for old, new in (
        ('publication=current_git("rev-parse","HEAD")', 'publication=certified_tip'),
        ('mod.published(repo,publication,', 'check_preverified_publication(repo,publication,'),
        ('try:\n    q=subprocess.run(inner+[command],input=transport_body,capture_output=True,timeout=120)',
         'try:\n    check_preverified_publication(repo,publication,pinned_members)\n'
         'except (OSError, ValueError, subprocess.SubprocessError) as problem:\n'
         '    print(json.dumps({"standby_no_dispatch":"publication recheck before sending: " + type(problem).__name__}));sys.exit(0)\n'
         'if time.monotonic()>=cutoff:\n'
         '    print(json.dumps({"standby_no_dispatch":"fresh device receipt expired at send boundary"}));sys.exit(0)\n'
         'try:\n    q=subprocess.run(inner+[command],input=transport_body,capture_output=True,timeout=120)'),
    ):
        occurrences = 2 if old.startswith('mod.published') else 1
        need(source.count(old) == occurrences, "frozen publication or send anchor changed")
        source = source.replace(old, new)
    return compile(source, "<qualified standby runner with earlier publication check>", "exec")


def child_reservation(batch, nodes, dirs):
    need(re.fullmatch(r'[a-f0-9]{32}', dirs), 'original batch invocation is not bound')
    need(1 <= len(nodes) <= 8 and len(set(nodes)) == len(nodes)
         and all(re.fullmatch(r'f[0-9]{2}r[0-9]+n[0-9]{2}', node) for node in nodes), 'invalid exact child nodes')
    seen = set()
    for node in nodes:
        for path in JOURNAL.glob(f'{node}-*-*.json'):
            if not re.fullmatch(rf'{node}-(?:1103606|1194900)\.[0-9]+-[012]\.json', path.name):
                continue
            marker = json.loads(path.read_bytes())
            if marker.get('wrapper_invocation') != dirs:
                continue
            need(marker['node'] == node and marker['batch'] == batch and marker['replay_allowed'] is False
                 and 0 <= time() - marker['recorded_epoch'] <= 45, 'original node reservation differs or aged')
            need(node not in seen, 'ambiguous original node reservation')
            seen.add(node)
    need(seen == set(nodes) and not (JOURNAL / f'{dirs}-outcome.json').exists(), 'missing original atomic reservation')


def prior_to_device(batch, invocation):
    need(os.environ.get("HYGON_STANDBY_VIA_COMPUTE") == "1", "ordinary renewal must use its original compute bridge")
    nodes = os.environ.get('HYGON_STANDBY_ONLY_NODES', '').split()
    child_reservation(batch, nodes, invocation)
    with (JOURNAL / f'{invocation}-publication-refill-once.json').open('x', encoding='utf8') as marker:
        json.dump(dict(wrapper_invocation=invocation, batch=batch, nodes=nodes, replay_allowed=False), marker)
        marker.flush()
        os.fsync(marker.fileno())
    try:
        action = render()
        state = precheck()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"standby_no_dispatch": "publication precheck before device admission: " + type(error).__name__}))
        return
    os.chdir(REPO.parent)
    check = lambda repo, tip, paths: verify_bound(state, repo, tip, paths)
    exec(action, dict(__name__="__main__", certified_tip=state["head"], check_preverified_publication=check,
                      pinned_members=ALL))


def reserve_after_publication_timeout(journal, node, step, intent, base, proof):
    if proof['NODES'].get(node) != step:
        return base['reserve_node'](journal, node, step, intent)
    previous = journal / f"{proof['WRAPPER']}-outcome.json"
    if not previous.is_file() or node not in proof['confirmed_nodes'](journal, json.loads(previous.read_bytes())):
        return base['reserve_node'](journal, node, step, intent)
    evidence = dict(proof_source_SHA256=ORIGINAL_TIMEOUT_PROOF_SHA256, original_wrapper=proof['WRAPPER'],
                    original_outcome_SHA256=proof['OUTCOME_SHA'])
    for slot in (1, 2):
        marker = journal / f'{node}-{step}-{slot}.json'
        try:
            base['store'](marker, dict(intent, original_prepublication_timeout=evidence))
        except FileExistsError:
            prior = json.loads(marker.read_bytes())['wrapper_invocation']
            need(re.fullmatch(r'[a-f0-9]{32}', prior), 'invalid retained invocation')
            outcome = journal / f'{prior}-outcome.json'
            if not outcome.is_file():
                return False
            old = json.loads(outcome.read_bytes())
            if old['wrapper_invocation'] != prior or node not in old['terminal_ready_nodes'] or old['no_GPU_send_confirmed'] is not True:
                return False
        else:
            return True
    return False


def run_batch(batch, dispatch):
    need(digest(FROZEN.read_bytes()) == FROZEN_SHA256, "frozen ordinary journal controller changed")
    need(digest(ORIGINAL_TIMEOUT_PROOF.read_bytes()) == ORIGINAL_TIMEOUT_PROOF_SHA256,
         "original confirmed prepublication-timeout verifier changed")
    base = runpy.run_path(str(FROZEN))
    proof = runpy.run_path(str(ORIGINAL_TIMEOUT_PROOF))
    original = base['run']

    def route(argv, *args, **kwargs):
        if argv == ["pwsh", "-NoProfile", "-File", str(REFILL)]:
            nodes = kwargs['env']['HYGON_STANDBY_ONLY_NODES'].split()
            invocations = []
            for node in nodes:
                current = {row['wrapper_invocation'] for path in JOURNAL.glob(f'{node}-*-*.json')
                           if re.fullmatch(rf'{node}-(?:1103606|1194900)\.[0-9]+-[012]\.json', path.name)
                           and (row := json.loads(path.read_bytes()))['batch'] == batch
                           and 0 <= time() - row['recorded_epoch'] <= 45}
                invocations.append(current)
            possible = {item for item in set.intersection(*invocations) if not (JOURNAL / f'{item}-outcome.json').exists()}
            need(len(possible) == 1, "original journal invocation is ambiguous")
            invocation, = possible
            child_reservation(batch, nodes, invocation)
            argv = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--refill",
                    "--child-batch", str(batch), "--invocation", invocation]
            kwargs['timeout'] = 250
        return subprocess.run(argv, *args, **kwargs)

    original.__globals__['subprocess'] = SimpleNamespace(run=route, TimeoutExpired=subprocess.TimeoutExpired)
    original.__globals__['reserve_node'] = lambda journal, node, step, intent: reserve_after_publication_timeout(
        journal, node, step, intent, base, proof)
    return original(batch, dispatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--batch', type=int)
    action.add_argument('--refill', action='store_true', help=argparse.SUPPRESS)
    action.add_argument('--check', action='store_true')
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--child-batch', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--invocation', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.refill:
        prior_to_device(args.child_batch, args.invocation or '')
    elif args.check:
        render()
        print(json.dumps(dict(compile=True, members=len(ALL), early_git_timeout_seconds=58, remote_calls=0)))
    else:
        print(json.dumps(run_batch(args.batch, args.dispatch), sort_keys=True))


if __name__ == '__main__':
    main()
