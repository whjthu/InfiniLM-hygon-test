"""Pass the two guarded T07 stages through one authenticated, finite SSH process."""
import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
from queue import Empty, Queue
import re
import runpy
import shlex
import subprocess
from threading import Thread
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO / "local-only"
GPU_REQUEST = "8dcecfc1023c4bb5bc05f4bccb850b1f"
DEVICE_REQUEST = "e8e817a8274646dbadc3a026c985f33e"
PREVIOUS_DEVICE = "f6b4bc0aeb2244fa8a22cdc831aafa05"
ORIGINAL_REQUEST = "42f59a4f4f004d6db5ac2c4560afc034"
PARENT, NODE = "1103606", "f13r2n07"
PYTHON = "/usr/local/python3.12/bin/python3"
PREVIOUS_LAUNCH = REPO / "code/bw1000/training80-t07-2429-durable-device-candidate-v1/launch.py"
PREVIOUS_LAUNCH_SHA256 = "df713ab41dbeea42d2d03df7d7cc2e747be1b8f63716a6fec163b5104750ef1a"
PREVIOUS_OUTCOME = PRIVATE / f"training80-T05-refill-transfer-{GPU_REQUEST}"
PREVIOUS_OUTCOME_SHA256 = "0d761685f3d942ae768c382f097a92e5181a525639e0ecaacba31cf11d85f061"
PREVIOUS_ACCOUNT = PRIVATE / "training80-T07-2631-zero-candidate-authority-no-send-2409b66d93ca4ec690e5854e89daf659.json"
PREVIOUS_ACCOUNT_SHA256 = "0149c198002b56e7e16530e418133721345a39a4fb9953b82a9c8b82904e2875"


def digest(raw):
    return sha256(raw).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def package():
    need(digest(PREVIOUS_LAUNCH.read_bytes()) == PREVIOUS_LAUNCH_SHA256, "prior confirmed guard source differs")
    previous = runpy.run_path(str(PREVIOUS_LAUNCH))
    old, _, _ = previous["package"]()
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    need(claim["GPU_request"] == GPU_REQUEST and claim["device_request"] == DEVICE_REQUEST
         and set(claim["files"]) == {"launch.py", "observe.py", "session.py"}
         and all(digest((HERE / name).read_bytes()) == pin for name, pin in claim["files"].items()),
         "same-session source files or claim identity differ")
    observer, session = ((HERE / name).read_bytes() for name in ("observe.py", "session.py"))
    compile(observer, "<fixed device worker wrapper>", "exec")
    compile(session, "<two-stage authenticated host>", "exec")

    # Reuse the existing exact GPU guard for this replacement observation ID and label.
    render = previous["render"]
    render.__globals__["NEW_ADMISSION"] = DEVICE_REQUEST
    bridge = render(digest(observer), old)
    guard_old_label = b"training80-t07-f13r2n07-candidate-"  # retained by the wire guard and device wrapper
    need(guard_old_label in bridge and DEVICE_REQUEST.encode() in bridge, "frozen guard device scope differs")
    return previous, old, observer, session, bridge


def earlier_exact_unsent(*, keep_new_absent=True):
    status_bytes = (PREVIOUS_OUTCOME / "outcome.json").read_bytes()
    account_bytes = PREVIOUS_ACCOUNT.read_bytes()
    need(digest(status_bytes) == PREVIOUS_OUTCOME_SHA256 and digest(account_bytes) == PREVIOUS_ACCOUNT_SHA256,
         "previous local no-send or precise accounting changed")
    status, account = json.loads(status_bytes), json.loads(account_bytes)["observed"]
    previous_step, = account["zero_exact_bounds"]
    need(status["request"] == account["request"] == GPU_REQUEST
         and status["authority_received"] is False and status["payload_send_attempted"] is False
         and status["payload_flushed"] is False and status["remote_payload_received"] is False
         and not (PREVIOUS_OUTCOME / "send-token.json").exists()
         and previous_step[:4] == ["1103606.2631", "training80-t07-f13r2n07-candidate-f6b4bc0aeb22", "COMPLETED", "0:0"]
         and not any(n["exists"] or n["symlink"] for n in account["namespaces"].values()),
         "previous attempt is not proven to have stopped before GPU send")
    new_tx = PRIVATE / f"training80-T07-2429-one-session-{DEVICE_REQUEST}"
    need(not keep_new_absent or not new_tx.exists(), "one-session scope already invoked")


def make_payload(publication, bridge, certificate):
    path = REPO / "code/bw1000/fleet-dcu-standby-v2"
    worker, controller = ((path / name).read_bytes() for name in ("worker.py", "controller.py"))
    claim = json.loads((path / "CLAIM.json").read_bytes())
    need(digest(worker) == claim["worker_SHA256"] and digest(controller) == claim["controller_SHA256"],
         "published worker or controller differs")
    binding = dict(request=GPU_REQUEST, publication=publication, seconds=10800,
                   nodes={NODE: {"parent": PARENT, "pci": certificate["PCIs"]}}, worker=worker, claim=claim,
                   admission_SHA256=certificate["payload_SHA256"])
    entry = ("INPUT=" + repr(binding) + "\n").encode() + controller
    return json.dumps(dict(request=GPU_REQUEST, entry=b64encode(entry).decode(), entry_SHA256=digest(entry),
          bridge=b64encode(bridge).decode(), bridge_SHA256=digest(bridge), t07_successor_device=certificate),
          sort_keys=True).encode()


def same_process(argv, publication, observer, session, bridge, previous, old, args):
    save, json_bytes = (previous[name] for name in ("save", "json_bytes"))
    argv = [argv[0], "-v", *argv[1:], shlex.join([
        PYTHON, "-I", "-S", "-B", "-u", "-c", session.decode(), b64encode(observer).decode(),
        digest(observer), digest(session), digest(bridge)])]
    dest = PRIVATE / f"training80-T07-2429-one-session-{DEVICE_REQUEST}"
    dest.mkdir(mode=0o700)
    scope = dict(device_request=DEVICE_REQUEST, GPU_request=GPU_REQUEST)
    audit = dict(**scope, zero_start_written=False, host_authority=False, candidate_received=False,
                 GPU_authority=False, payload_send_attempted=False, payload_flushed=False,
                 remote_payload_received=False, bridge_ack=False)
    save(dest / "intent.json", json_bytes(dict(**scope, publication=publication, predecessor_SHA256=args.sha,
         observer_SHA256=digest(observer), session_SHA256=digest(session), bridge_SHA256=digest(bridge), replay_allowed=False)))
    with (dest / "ssh-stderr.private").open("xb") as stderr:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        frames = Queue(maxsize=8)

        def collect_frames():
            for line in iter(lambda: process.stdout.readline(8194), b""):
                if len(line) > 8193 or not line.endswith(b"\n"):
                    frames.put(None)
                    return
                frames.put(line)
            frames.put(None)

        Thread(target=collect_frames, daemon=True).start()

        def next_frame(wait):
            raw = frames.get(timeout=max(.1, wait))
            now = time.monotonic()
            need(raw is not None and len(raw) <= 8193, "ended or unbounded host response")
            framed = json.loads(raw)
            if framed.get("stage") == "HOST_ERROR":
                audit["host_error"] = framed
                raise ValueError("authenticated host returned a bounded refusal")
            return framed, now

        def write(frame):
            process.stdin.write(json_bytes(frame))
            process.stdin.flush()

        try:
            authority, _ = next_frame(18)
            need(authority == dict(stage="HOST_AUTHORITY", **scope, observer_SHA256=digest(observer),
                                   session_SHA256=digest(session)), "unauthorized wrong host session scope")
            audit["host_authority"] = True
            old["predecessor"](args.terminal, args.sha, ready=True)
            previous["protected_jobs"]()
            started = time.monotonic()
            save(dest / "zero-start-token.json", json_bytes(dict(**scope, started_monotonic=started, replay_allowed=False)))
            audit["zero_start_written"] = True
            write(dict(stage="START_ZERO_GRES", **scope))
            frame, received = next_frame(40)
            candidate = frame["candidate"]
            certificate = candidate["certified_observation"]
            raw = json_bytes(candidate)
            pin = digest(raw)
            need(frame == dict(stage="DEVICE_CANDIDATE", **scope, candidate_SHA256=pin, candidate=candidate)
                 and candidate["request"] == DEVICE_REQUEST and candidate["gpu_request"] == GPU_REQUEST
                 and candidate["observer_SHA256"] == digest(observer)
                 and candidate["status"] == "ZERO_GRES_DEVICE_CANDIDATE" and candidate["current_probe_srun_rc"] == 0
                 and certificate["request_id"] == DEVICE_REQUEST and certificate["node"] == NODE
                 and certificate["parent"] == PARENT and certificate["active_steps"] == []
                 and certificate["kfd_process_count"] == certificate["own_device_fd_process_count"] == 0,
                 "independent physical candidate failed current scope checks")
            cutoff = min(received + 25, started + 40)
            need(0 <= received - started < 40 and time.monotonic() < cutoff, "current device candidate exceeded monotonic freshness")
            save(dest / "candidate.json", raw)
            save(dest / "received-anchor.json", json_bytes(dict(**scope, started_monotonic=started,
                 received_monotonic=received, deadline_monotonic=cutoff, candidate_SHA256=pin)))
            audit["candidate_received"] = True
            earlier_exact_unsent(keep_new_absent=False)
            previous["protected_jobs"]()
            packet = make_payload(publication, bridge, certificate)
            sizes = dict(candidate_SHA256=pin, payload_SHA256=digest(packet), payload_bytes=len(packet))
            need(0 < len(packet) <= 262144 and time.monotonic() < cutoff, "payload is unbounded or candidate expired")
            write(dict(stage="OFFER_GPU_PAYLOAD", **scope, **sizes))
            accepted, _ = next_frame(cutoff - time.monotonic())
            need(accepted == dict(stage="GPU_AUTHORITY", **scope, **sizes)
                 and time.monotonic() < cutoff, "no matching host authority while the device candidate was fresh")
            audit["GPU_authority"] = True
            save(dest / "GPU-send-token.json", json_bytes(dict(**scope, **sizes, monotonic=time.monotonic(), replay_allowed=False)))
            need(time.monotonic() < cutoff, "original candidate expired before the first GPU payload byte")
            audit["payload_send_attempted"] = True
            process.stdin.write(packet + b"\n")
            process.stdin.flush()
            audit["payload_flushed"] = True
            got, _ = next_frame(10)
            need(got == dict(stage="HOST_RECEIVED_GPU_PAYLOAD", **scope, **sizes), "no authenticated original payload receipt")
            audit["remote_payload_received"] = True
            reply, _ = next_frame(40)
            need(reply["stage"] == "BRIDGE_REPLY" and reply["device_request"] == DEVICE_REQUEST
                 and reply["GPU_request"] == GPU_REQUEST, "bridge reply scope differs")
            save(dest / "bridge-reply.json", json_bytes(reply))
            audit["bridge_ack"] = reply["rc"] == 0 and isinstance(reply["reply"], dict) and \
                reply["reply"].get("bridge_launch_pending") is True and reply["reply"].get("request") == GPU_REQUEST
        except (Empty, OSError, ValueError, KeyError, TypeError) as error:
            audit["error_type"] = type(error).__name__
            audit["detail_SHA256"] = digest(str(error).encode("utf8", "replace"))
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                audit["rc"] = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                audit["rc"] = process.wait(timeout=3)
    raw = (dest / "ssh-stderr.private").read_bytes()
    audit.update(ssh_authenticated=b"Authenticated to" in raw, ssh_command_sent=b"Sending command:" in raw,
                 ssh_exec_accepted=b"exec request accepted" in raw, stderr_SHA256=digest(raw))
    pin = save(dest / "outcome.json", json_bytes(audit))
    print(json.dumps(dict(outcome=audit, outcome_SHA256=pin), sort_keys=True), flush=True)
    return 0 if audit["bridge_ack"] and audit["rc"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publication")
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    previous, old, observer, session, bridge = package()
    previous["protected_jobs"]()
    if args.check:
        print(json.dumps(dict(offline_check=True, GPU_request=GPU_REQUEST, device_request=DEVICE_REQUEST,
                              bridge_SHA256=digest(bridge), remote_calls=0), sort_keys=True))
        return 0
    need(args.publication and re.fullmatch("[a-f0-9]{40}", args.publication)
         and args.publication != "0" * 40 and args.terminal and args.sha and re.fullmatch("[a-f0-9]{64}", args.sha),
         "verified publication and fresh actual old terminal evidence required")
    earlier_exact_unsent()
    old["predecessor"](args.terminal, args.sha, ready=True)
    helper = runpy.run_path(str(PRIVATE / "run-minicpm-sdpa-target-asset-preflight-original.py"))
    return same_process(helper["ssh"](REPO.parent), args.publication, observer, session, bridge, previous, old, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(error_type=type(error).__name__,
                              detail_SHA256=digest(str(error).encode("utf8", "replace")))))
        raise SystemExit(1)
