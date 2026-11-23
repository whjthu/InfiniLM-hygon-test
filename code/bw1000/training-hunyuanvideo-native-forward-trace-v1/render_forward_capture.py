"""Create the opt-in forward diagnostic from the measured native backward collector."""

from hashlib import sha256
import ast


BACKEND_SHA256 = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"
COLLECTOR_SHA256 = "3176cbec0e421ce84cf495360aee11d09b6c7332068977c565341ff1f9af11fc"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("The frozen diagnostic source anchor differs")
    return text.replace(old, new, 1)


def render(backend):
    if sha256(backend).hexdigest() != BACKEND_SHA256:
        raise ValueError("Original native HunyuanVideo backend differs")
    source = backend.decode()
    head = "        def train_step(global_step: int) -> dict[str, float]:\n"
    init = (
        "        from contextlib import nullcontext\n"
        "        from .hunyuanvideo_forward_trace import STEP as TRACE_STEP, trace_from_environment\n"
        "        forward_trace = trace_from_environment(\n"
        "            torch, output_dir, warmup=warmup_steps, measured=max_steps,\n"
        '            restored_step=completed_global_step, recovery=config.get("validate_recovery", False),\n'
        "        )\n\n"
    )
    start = (
        "            forward_started = perf_counter()\n"
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
    )
    stop = "                loss = torch.nn.functional.mse_loss(prediction, target)\n"
    if source.count(start) != 1 or source.count(stop) != 1:
        raise ValueError("The original forward timing or loss anchor differs")
    lo = source.index(start) + len("            forward_started = perf_counter()\n")
    hi = source.index(stop, lo) + len(stop)
    original = source[lo:hi]
    wrapped = (
        "            capture = (forward_trace.capture()\n"
        "                       if forward_trace is not None and global_step == TRACE_STEP else nullcontext())\n"
        "            with capture:\n"
        + "".join("    " + line for line in original.splitlines(keepends=True))
    )
    modified = replace_once(replace_once(source, head, init + head), original, wrapped)
    if replace_once(replace_once(modified, wrapped, original), init + head, head) != source:
        raise ValueError("Forward diagnostic rewrote model math")
    ast.parse(modified)
    return modified.encode()


def render_observer(collector, sdpa_metadata):
    if sha256(collector).hexdigest() != COLLECTOR_SHA256:
        raise ValueError("The measured collector revision differs")
    source = collector.decode()
    source = source.replace("backward", "forward").replace("Backward", "Forward").replace("BACKWARD", "FORWARD")
    for old, new in (
        ("def __init__(self, profiler, output, *, training_step):",
         "def __init__(self, torch, output, *, training_step):\n        profiler = torch.profiler"),
        ("        self.profiler = profiler", "        self.profiler = profiler\n        self.functional = torch.nn.functional"),
        ("        with profiler.profile(activities=self.activities, record_shapes=False, profile_memory=False,",
         "        with audit_sdpa_descriptors(self.functional) as descriptors:\n"
         "            with profiler.profile(activities=self.activities, record_shapes=False, profile_memory=False,"),
        ("                              with_stack=False, with_flops=False) as capture:\n"
         "            with profiler.record_function(MARKER):\n                yield",
         "                                  with_stack=False, with_flops=False) as capture:\n"
         "                with profiler.record_function(MARKER):\n                    yield"),
        ("        raw = self.directory", '        write_json(self.directory / "sdpa-descriptors.json", descriptors)\n'
                                        "        raw = self.directory"),
        ("return ForwardTrace(torch.profiler, output, training_step=STEP)",
         "return ForwardTrace(torch, output, training_step=STEP)"),
    ):
        source = replace_once(source, old, new)
    # The audited metadata recorder runs only inside the selected profile window.
    source += "\n\n" + sdpa_metadata.decode()
    ast.parse(source)
    return source.encode()
