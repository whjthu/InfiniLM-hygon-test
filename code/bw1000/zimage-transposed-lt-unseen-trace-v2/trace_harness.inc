def trace_size(directory, *, live, state):
    total = 0
    names = set()
    with os.scandir(directory) as entries:
        for number, entry in enumerate(entries, 1):
            spec.need(number <= 20, "UNEXPECTED_TRACE_SCOPE")
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                if not live:
                    raise
                state["vanished_live_members"] += 1
                continue
            spec.need(stat.S_ISREG(info.st_mode), "UNEXPECTED_TRACE_SCOPE")
            names.add(entry.name)
            total += info.st_size
            spec.need(total <= TRACE_LIMIT, "TRACE_DIRECTORY_BUDGET")
    if not live:
        spec.need({"tool.stdout.private", "tool.stderr.private"} <= names, "MISSING_TERMINAL_TOOL_LOG")
        state["terminal_members"] = len(names)
        state["terminal_bytes"] = total
    return total


def trace_error(error, phase):
    evidence = {"phase": phase, "type": type(error).__name__}
    if isinstance(error, OSError):
        evidence["errno"] = error.errno
        if error.filename:
            leaf = Path(os.fsdecode(error.filename)).name
            if leaf in {"hipprof", "worker-output", "trace.json", "candidate-witness.json",
                        "tool.stdout.private", "tool.stderr.private", "null"}:
                evidence["known_basename"] = leaf
    return evidence


def stop_trace_child(child):
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


def in_trace(command, directory, deadline, state):
    state.update(phase="tool_fingerprint", child_PID=None, live_scans=0, vanished_live_members=0)
    child = None
    try:
        fingerprint(TRACER, TRACER_SHA)
        base = [str(TRACER), "--hip-trace", "--trace-off", "--show-pid", "--output-type", "0",
                "-d", str(directory), "-o", str(directory / "trace")]
        state["phase"] = "log_open"
        with (directory / "tool.stdout.private").open("xb") as out, (directory / "tool.stderr.private").open("xb") as err:
            state["phase"] = "profiler_exec"
            child = subprocess.Popen([*base, *command], start_new_session=True, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err)
            state.update(child_PID=child.pid, phase="profiler_poll")
            while child.poll() is None:
                state["phase"] = "live_scan"
                state["live_scans"] += 1
                trace_size(directory, live=True, state=state)
                state["phase"] = "deadline"
                spec.need(time.time() < deadline, "NATIVE_TRACE_TIMEOUT")
                time.sleep(0.15)
                state["phase"] = "profiler_poll"
        state["phase"] = "terminal_scan"
        trace_size(directory, live=False, state=state)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        state["failure"] = trace_error(error, state["phase"])
        raise
    finally:
        if child is not None and child.poll() is None:
            primary_error = sys.exc_info()[0]
            try:
                stop_trace_child(child)
            except (OSError, subprocess.SubprocessError) as cleanup_error:
                state["cleanup_failure"] = trace_error(cleanup_error, "profiler_cleanup")
                if primary_error is None:
                    raise
    state.update(phase="returned", returncode=child.returncode)
    return child.returncode
