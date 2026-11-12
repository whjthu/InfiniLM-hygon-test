"""Run the frozen public R454 server once and compare the pinned history alias OFF and ON."""

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from pilot_inputs import compact, require, resolve, sha
from pilot_observation import capture, evaluate, wire_identity_and_first_text
from alias_policy import SCHEDULE
from alias_summary import summarize

SOURCE = Path("/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1")
SOURCE_SHA = "bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a"
OUTPUT = Path("/pilot-output")
RUN = "i03-deep-offoff-repeatability-v1"
NATIVE = "5e6793ad220e2bb6501fadb3eae70f0b5270565759f2c6907997809ef1be3537"


def publish(path, data):
    raw = compact(data) + b"\n"
    tmp = path.with_suffix(".writing")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def emit(stage, **data):
    print(compact(dict(kind="i03_r454_http", stage=stage, **data)).decode(), flush=True)


def wait_for(path, process, end, predicate=lambda value: True):
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("R454 frontend exited before " + path.name)
        if (path.parent / "FAILED.json").exists():
            raise RuntimeError("R454 reported failure before " + path.name)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            time.sleep(.1)
            continue
        require(len(raw) <= 8 * 1024**2, "oversize private record")
        record = json.loads(raw)
        if predicate(record):
            return record
        time.sleep(.1)
    raise TimeoutError("bounded wait for " + path.name)


def wait_http(port, process, end):
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("R454 frontend exited before health")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=.4)
        try:
            connection.request("GET", "/health")
            with connection.getresponse() as response:
                if response.status == 200 and json.loads(response.read(1024)) == {"status": "healthy"}:
                    return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(.1)
    raise TimeoutError("R454 HTTP socket did not become healthy")


def run_requests(selected, *, port, process, folder, identity, deadline, parser, grader, planned):
    for i, (entry, body, case) in enumerate(selected):
        slot = planned[i]
        if time.monotonic() >= deadline:
            raise TimeoutError("budget exhausted before the next fixed request")
        stem = OUTPUT / ("request-" + str(i))
        wire, clocks = stem.with_suffix(".wire"), stem.with_suffix(".clock.jsonl")
        slot["state"] = "DISPATCH_INTENT"
        publish(OUTPUT / "progress.json", planned)
        emit("request_enter", slot=i, prompt_tokens=entry["prompt_tokens"], budget=entry["output_budget"])
        try:
            status = capture(port, body, deadline, wire, clocks)
            slot["HTTP_status"] = status
            if status != 200:
                slot["state"] = "HTTP_REJECTED"
                publish(OUTPUT / "progress.json", planned)
                emit("request_terminal", slot=i, state=slot["state"], HTTP_status=status)
                raise RuntimeError("HTTP rejected an original fixed request")
            wire_bytes = wire.read_bytes()
            reads = [json.loads(line) for line in clocks.read_bytes().splitlines()]
            request_id, _ = wire_identity_and_first_text(wire_bytes, reads)
            def contains(record):
                require(record["identity"] == identity, "foreign R454 owner record")
                return any(row["request_id"] == request_id and "final_metadata" in row
                           for row in record["requests"])
            owner = wait_for(folder / "owner.json", process, min(deadline, time.monotonic() + 30), contains)
            matches = [row for row in owner["requests"] if row["request_id"] == request_id]
            require(len(matches) == 1 and matches[0]["r454_index"] == i, "request order or identity differs")
            result = evaluate(entry, case, wire_bytes, reads, matches[0], decode_sse=parser, grade=grader)
            publish(stem.with_suffix(".summary.json"), result)
            slot.update(state="RECORDED", score=result["score"],
                        wire_SHA256=result["wire_SHA256"], backend_TTFT_s=result["backend_TTFT_s"],
                        socket_first_text_s=result["socket_first_text_s"],
                        backend_non_eos_decode_tok_s=result["backend_non_eos_decode_tok_s"],
                        generated_including_EOS=result["generated_including_EOS"],
                        backend_non_eos_token_count=result["backend_non_eos_token_count"],
                        native_forwards=result["native_forwards"])
            emit("request_terminal", slot=i, mode=SCHEDULE[i][1], **{k:v for k,v in slot.items() if k != "case_id"})
        except Exception as error:
            if slot["state"] != "HTTP_REJECTED":
                slot["state"] = "UNVERIFIED"
            slot["error_type"] = type(error).__name__
            emit("request_terminal", slot=i, state=slot["state"], error_type=slot["error_type"])
            raise
        finally:
            publish(OUTPUT / "progress.json", planned)
    require(all(slot["state"] == "RECORDED" for slot in planned), "one or more request slots lacks verified evidence")


def service_bootstrap(root, pins):
    return (
        "import hashlib,sys;from pathlib import Path;"
        f"p=Path({str(root)!r});pins={pins!r};"
        "assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in pins.items());"
        "sys.path.insert(0,str(p));"
        "import alias_policy,diagnostic_binding;"
        "diagnostic_binding.install();"
        "from public_deep_model.request_policy import RequestPolicy;"
        "from public_deep_model.worker import RequestPolicy as WorkerPolicy;"
        "assert WorkerPolicy is RequestPolicy;"
        "alias_policy.install(RequestPolicy);"
        "assert WorkerPolicy.forward.__module__=='alias_policy';"
        "from public_deep_model.cli import main;main()"
    )


def run(epoch, bundle, parser, grader):
    remaining = epoch + 7200 - time.time()
    require(math.isfinite(remaining) and 7000 < remaining <= 7200, "unexpected initial service budget")
    service_end = time.monotonic() + remaining - 210
    native_end = service_end + 120
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(step.startswith("1087656.") and step.split(".")[1].isdigit()
            and socket.gethostname().split(".")[0] == "f12r3n12", "unexpected Slurm ownership")
    cfg = json.loads(bundle["service_config.json"])
    require(cfg["host"] == "127.0.0.1" and cfg["port"] == 43888
            and cfg["runtime_directory"] == str(OUTPUT) and cfg["node"] == "f12r3n12"
            and cfg["Native_SHA256"] == NATIVE and cfg["Native_seconds"] == 7110
            and cfg["service_seconds"] == 6990 and cfg["experimental_profile"] is None,
            "unexpected service lifecycle config")
    spec = json.loads(bundle["input_binding.json"])
    inputs = resolve(Path("/refactor"), spec)
    selected = tuple(inputs[case] for case, _ in SCHEDULE)
    from public_deep_model.cli import verify_package
    require(verify_package(SOURCE_SHA) == SOURCE, "unexpected R454 provider location")
    planned = [dict(case_id=entry["case_id"], suite=entry["suite"], state="PLANNED")
               for entry, _, _ in selected]
    publish(OUTPUT / "progress.json", planned)
    with socket.socket() as available:
        available.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        available.bind((cfg["host"], cfg["port"]))
    config_path = OUTPUT / "service-config.json"
    with config_path.open("xb") as stream:
        stream.write(bundle["service_config.json"])
    config_sha = sha(bundle["service_config.json"])
    identity = dict(run=RUN, actual_step=step, source_manifest_SHA256=SOURCE_SHA,
                    provider_config_SHA256=config_sha, gpu_start_epoch=epoch, Native_seconds=7110)
    folder = OUTPUT / RUN
    files = ("alias_policy.py", "diagnostic_binding.py", "pilot_inputs.py")
    for name in files:
        with (OUTPUT / name).open("xb") as f:
            f.write(bundle[name])
    child = service_bootstrap(OUTPUT, {name: hashlib.sha256(bundle[name]).hexdigest() for name in files})
    command = [sys.executable, "-B", "-u", "-c", child, "--serve",
               "--config", str(config_path), "--config-sha", config_sha, "--source-sha", SOURCE_SHA,
               "--run", RUN, "--actual-step", step, "--gpu-start-epoch", str(epoch)]
    emit("container_admitted", step=step, planned=len(planned), config_SHA256=config_sha)
    with (OUTPUT / "service.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            ready = wait_for(folder / "MODEL_READY.json", process, service_end - 120)
            require(ready["identity"] == identity and ready["Model_READY"] is True
                    and ready["Native_SHA256"] == NATIVE and ready["Native_internal_owner_count"] == 8
                    and ready["fixed_profile"]["fixed_mode"] == 0, "unverified real R454 readiness")
            engine = wait_for(folder / "ready-engine.json", process, service_end - 120)
            require(engine["identity"] == identity and engine["loaded_key_count"] == 67569
                    and engine["capacity"] == 32768 and engine["rows"] == 1, "loaded checkpoint evidence differs")
            wait_http(cfg["port"], process, min(service_end - 120, time.monotonic() + 30))
            emit("model_ready", step=step, loaded_key_count=engine["loaded_key_count"])
            run_requests(selected, port=cfg["port"], process=process, folder=folder, identity=identity,
                         deadline=service_end - 10, parser=parser, grader=grader, planned=planned)
        finally:
            if process.poll() is None and folder.exists():
                from public_deep_model.protocol import publish_once
                try:
                    publish_once(folder / "STOP.json", dict(identity=identity, normal_stop=True))
                except FileExistsError:
                    require(json.loads((folder / "STOP.json").read_bytes())["identity"] == identity,
                            "foreign stop intent")
            try:
                rc = process.wait(timeout=max(0.01, native_end - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                emit("forced_cleanup", step=step)
                raise
            emit("service_exit", step=step, exit_code=rc)
    closed = json.loads((folder / "closed-engine.json").read_bytes())
    require(rc == 0 and closed["identity"] == identity and closed["normal"] is True
            and closed["engine_destroy_returned_zero"] is True
            and closed["observed_all8_threads_exited"] is True, "normal R454 destroy not verified")
    owner = json.loads((folder / "owner.json").read_bytes())
    require(owner["identity"] == identity and len(owner["requests"]) == len(planned),
            "unexpected actual request count")
    artifact_names = ["progress.json", "service-config.json", "service.log"] + [
        f"request-{i}{suffix}" for i in range(len(planned)) for suffix in
        (".wire", ".clock.jsonl", ".summary.json")]
    artifact_names += [RUN + "/" + name for name in
                       ("owner.json", "MODEL_READY.json", "ready-engine.json", "closed-engine.json")]
    requests = [json.loads((OUTPUT / f"request-{i}.summary.json").read_bytes())
                for i in range(len(planned))]
    summary = summarize(owner["requests"], requests)
    publish(OUTPUT / "alias-summary.json", summary)
    artifact_names += ["alias-summary.json", *files]
    receipts = {name: sha((OUTPUT / name).read_bytes()) for name in artifact_names}
    emit("complete", step=step, planned=len(planned), recorded=len(planned),
         normal_destroy=True, valid_timing_pairs=summary["valid_timing_pairs"], artifacts_SHA256=receipts)


def main(bundle, parser, grader, epoch):
    try:
        run(epoch, bundle, parser, grader)
    except BaseException as error:
        # Leave complete output and errors in the private directory.
        import traceback
        with (OUTPUT / "pilot-error.log").open("x") as out:
            traceback.print_exc(file=out)
        emit("pilot_failed", error_type=type(error).__name__)
        return 1
    return 0
