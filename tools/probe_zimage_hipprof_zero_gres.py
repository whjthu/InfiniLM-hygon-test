"""Probe the original hipprof startup and directory scan on a zero-GRES step."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import time


TRACER = Path("/opt/dtk/hip/bin/hipprof")
TRACER_SHA256 = "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"
LIMIT = 16 << 20


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def interpreter(binary):
    with binary.open("rb") as stream:
        identity = stream.read(16)
        require(identity[:6] == b"\x7fELF\x02\x01", "not a little-endian ELF64 executable")
        header = struct.unpack("<HHIQQQIHHHHHH", stream.read(48))
        _, _, _, _, offset, _, _, _, entry_bytes, count, *_ = header
        require(entry_bytes == 56 and 0 < count <= 128, "unexpected program header table")
        for index in range(count):
            stream.seek(offset + index * entry_bytes)
            kind, _, start, _, _, size, _, _ = struct.unpack("<IIQQQQQQ", stream.read(56))
            if kind == 3:
                require(1 < size <= 256, "invalid PT_INTERP size")
                stream.seek(start)
                raw = stream.read(size)
                require(raw.endswith(b"\0") and raw.startswith(b"/"), "invalid ELF interpreter")
                return Path(raw[:-1].decode("ascii"))
    raise ValueError("dynamic ELF interpreter not present")


def scan_size(directory, *, live):
    total = entries = vanished = 0
    with os.scandir(directory) as children:
        for member in children:
            entries += 1
            require(entries <= 20, "trace directory member limit")
            try:
                metadata = member.stat(follow_symlinks=False)
            except FileNotFoundError:
                if not live:
                    raise
                vanished += 1
                continue
            require(stat.S_ISREG(metadata.st_mode), "non-regular trace member")
            total += metadata.st_size
    return total, vanished


def run_profiler(args, directory):
    directory.mkdir()
    stage = "profiler_exec"
    child = None
    audit = {"pid_observed": False, "vanished_during_live_scan": 0, "running_scans": 0}
    try:
        with (directory / "stdout.private").open("xb") as out, (directory / "stderr.private").open("xb") as err:
            child = subprocess.Popen([str(TRACER), *args], cwd="/", stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, start_new_session=True)
            audit["pid_observed"] = True
            stage = "live_directory_scan"
            deadline = time.monotonic() + 20
            while child.poll() is None:
                total, vanished = scan_size(directory, live=True)
                audit["vanished_during_live_scan"] += vanished
                audit["running_scans"] += 1
                require(total <= LIMIT, "trace byte limit")
                require(time.monotonic() < deadline, "profiler timeout")
                time.sleep(0.15)
        stage = "final_directory_scan"
        size, _ = scan_size(directory, live=False)
        require(size <= LIMIT, "terminal trace byte limit")
        audit.update(stage="returned", returncode=child.returncode, bytes=size)
    except (OSError, ValueError) as error:
        name = Path(error.filename).name if isinstance(error, OSError) and error.filename else None
        audit.update(stage=stage, error_type=type(error).__name__, errno=getattr(error, "errno", None),
                     error_basename=name)
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)
    audit["private_log_bytes"] = {
        n: p.stat().st_size if (p := directory / n).is_file() else None for n in ("stdout.private", "stderr.private")
    }
    audit["trace_json_present"] = (directory / "trace.json").is_file()
    return audit


def digest_file(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    require(os.environ.get("SLURM_STEP_GPUS", "") in ("", "(null)")
            and all(os.environ.get(key) == "" for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")),
            "diagnostic requires no GPU allocation and empty device visibility")
    require(digest_file(TRACER) == TRACER_SHA256, "hipprof differs from original pinned executable")
    loader = interpreter(TRACER)
    loader_exists = loader.is_file()
    help_result = run_profiler(["-h"], args.result_directory / "help")
    trace_dir = args.result_directory / "empty_trace"
    trace_result = run_profiler(
        ["--hip-trace", "--trace-off", "--show-pid", "--output-type", "0",
         "-d", str(trace_dir), "-o", str(trace_dir / "trace"), "/bin/true"], trace_dir)
    output = {"schema": "O01.hipprof.zero_gres_start_boundary.v1", "step": os.environ.get("SLURM_STEP_ID"),
              "tracer_SHA256": TRACER_SHA256, "ELF_interpreter": str(loader),
              "ELF_interpreter_present": loader_exists,
              "ELF_interpreter_SHA256": digest_file(loader) if loader_exists else None,
              "help": help_result, "no_GPU_child": trace_result,
              "actual_algorithm_or_GPU_qualification": False}
    with (args.result_directory / "summary.json").open("x", encoding="utf-8") as receipt:
        json.dump(output, receipt, sort_keys=True)
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
