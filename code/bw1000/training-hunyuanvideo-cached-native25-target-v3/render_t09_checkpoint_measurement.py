"""Expose the existing HunyuanVideo checkpoint-inclusive timing in the common manifest."""

import ast
from hashlib import sha256


ORIGINAL = {
    "measurement.py": "8a9bb4ec134b4545ef135db1bdef7874331234ed0a8a32b95bcac63462b1bb94",
    "hunyuanvideo_training_walltime.py": "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae",
}


def replace_once(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError("Frozen measurement anchor differs")
    return source.replace(original, replacement, 1)


def render(files):
    if set(files) != set(ORIGINAL) or any(sha256(files[name]).hexdigest() != digest for name, digest in ORIGINAL.items()):
        raise ValueError("Frozen measurement sources differ")
    consumer = files["measurement.py"].decode()
    consumer = replace_once(consumer, "    unit: str\n", "    unit: str\n    end_monotonic: float | None = None\n")
    consumer = replace_once(consumer,
        '    if not isinstance(value, Mapping) or set(value) != {"start_monotonic", "work_units", "unit"}:\n',
        '    fields = {"start_monotonic", "work_units", "unit"}\n'
        '    if not isinstance(value, Mapping) or set(value) not in (fields, fields | {"end_monotonic"}):\n')
    consumer = replace_once(consumer, "    return MeasurementWindow(start, units, unit)\n",
        '    end = None\n'
        '    if "end_monotonic" in value:\n'
        '        end = _finite_number(value["end_monotonic"], "end_monotonic")\n'
        '        if not start < end <= t1:\n'
        '            raise ValueError("measurement_window.end_monotonic is outside backend execution")\n'
        '    return MeasurementWindow(start, units, unit, end)\n')
    consumer = replace_once(consumer, "    duration = t3 - window.start_monotonic\n",
        '    end = window.end_monotonic\n'
        '    if end is not None:\n'
        '        if not t0 <= window.start_monotonic < end <= t1:\n'
        '            raise ValueError("backend measurement clock ordering is invalid")\n'
        '        metrics["measurement_version"] = "backend-measurement-v1"\n'
        '        metrics["measurement_scope"] = "current_invocation_post_warmup_through_backend_boundary"\n'
        '    duration = (t3 if end is None else end) - window.start_monotonic\n')

    backend = files["hunyuanvideo_training_walltime.py"].decode()
    backend = replace_once(backend,
        "        measurement_wall_seconds = perf_counter() - measurement_window_started\n",
        "        measurement_window_ended = perf_counter()\n"
        "        measurement_wall_seconds = measurement_window_ended - measurement_window_started\n")
    backend = replace_once(backend, "        return _build_training_result(\n", "        result = _build_training_result(\n")
    backend = replace_once(backend, "            recovery=recovery,\n        )\n    finally:\n",
        '            recovery=recovery,\n'
        '        )\n'
        '        if (not config.get("validate_recovery", False) and completed_global_step == 0\n'
        '                and len(measured) == max_steps):\n'
        '            result["measurement_window"] = {\n'
        '                "start_monotonic": measurement_window_started,\n'
        '                "end_monotonic": measurement_window_ended,\n'
        '                "work_units": len(measured),\n'
        '                "unit": "optimizer_steps",\n'
        '            }\n'
        '        return result\n'
        '    finally:\n')

    output = {"measurement.py": consumer, "hunyuanvideo_training_walltime.py": backend}
    for name, source in output.items():
        ast.parse(source, filename=name)
    return {name: text.encode() for name, text in output.items()}
