"""Bind the verified single-DCU owner controller to the unloaded lifecycle scope."""

import hashlib


HOST_SHA256 = "b274c78588a62e5011d01e963fa56705c839fab2b96214d1a815425ee207602a"
INNER_SHA256 = "8d6243d36c36be236e5f20dbf93c6092041761c2abf560c6ea0b355e6c6de7fb"


def bind(host, inside):
    if hashlib.sha256(host).hexdigest() != HOST_SHA256 or hashlib.sha256(inside).hexdigest() != INNER_SHA256:
        raise ValueError("original single-DCU controller differs")
    overlay = host.decode("utf8")
    replacements = [
        ('LABEL = "Deep-C4-causal-consumer-r483-runtime-owner-GPU-v2"',
         'LABEL = "Deep-C4-causal-consumer-r483-unloaded-lifecycle-GPU-v1"'),
        ('WAVE_INSIDE_R2323=Deep-C4-causal-consumer-r483/runtime-owner-v2/GPU_inside.sh',
         'WAVE_INSIDE_R2323=Deep-C4-causal-consumer-r483/unloaded-lifecycle-v1/GPU_inside.sh'),
        ('    prior = collect("Deep-C4-causal-consumer-r483-runtime-owner-CPU-v1")\n'
         '    require(prior and prior[0] == "1103606.157" and prior[2:4] == ["FAILED", "6:0"], "prior CPU run unsettled")',
         '    for phase, handle in (("gpu", "1103606.159"), ("post", "1103606.160")):\n'
         '        prior = collect("Deep-C4-causal-consumer-r483-runtime-owner-GPU-v2-" + phase)\n'
         '        require(prior and prior[0] == handle and prior[2:4] == ["COMPLETED", "0:0"], "prior device scope unsettled")'),
        ('    report.update(actual_step=handle, runtime_provenance_only=True, device_numerics_qualified=False)',
         '    from lifecycle import verify_unloaded\n'
         '    result = verify_unloaded(probe, api, binding, claim, PACKAGE)\n'
         '    report.update(actual_step=handle, runtime_provenance_only=False, device_numerics_qualified=False, lifecycle=result)\n'
         '    report.pop("entry_calls")\n'
         '    report.pop("runtime_created")'),
    ]
    for old, new in replacements:
        if overlay.count(old) != 1:
            raise ValueError("original controller anchor differs")
        overlay = overlay.replace(old, new, 1)
    original = b"/runtime-owner-v2/device_owner.py worker"
    if inside.count(original) != 1:
        raise ValueError("original container worker anchor differs")
    return overlay.encode("utf8"), inside.replace(original, b"/unloaded-lifecycle-v1/device_owner.py worker", 1)
