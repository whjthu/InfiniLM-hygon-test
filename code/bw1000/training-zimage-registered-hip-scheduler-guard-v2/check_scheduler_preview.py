"""Regression for the real FlowMatchScheduler snapshots in native registered replay."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


SOURCE = Path("/native-source/diffsynth/diffusion/flow_match.py")
FRAMEWORK = Path("/common-framework/generation/src")
SOURCE_SHA256 = "4000f836cd019673bbf5e0c34c34a3a8a15325c5fce615481cc81dc066656a87"


def main():
    sys.path.insert(0, str(FRAMEWORK))
    import torch
    from gencuda import zimage_native_record as register
    from gencuda import zimage_training_resume_checkpoint as resume

    if torch.cuda.is_available() or sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected zero GRES and original scheduler source")
    scheduler = runpy.run_path(str(SOURCE))["FlowMatchScheduler"]("Z-Image")
    scheduler.set_timesteps(1000, training=True)
    expected = resume.scheduler_spec(scheduler)
    before = resume.scheduler_spec(scheduler)
    try:
        before != expected
    except RuntimeError as error:
        if "Boolean value of Tensor" not in str(error):
            raise
    else:
        raise AssertionError("The original failure did not reproduce")

    original = Path(register.__file__).read_bytes()
    tree = ast.parse(original)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)
             and ast.unparse(node.value.func) == "resume._validate_scheduler"
             and [ast.unparse(arg) for arg in node.value.args] ==
             ["resume.scheduler_spec(model.pipe.scheduler)", "scheduler"]]
    if len(calls) != 1:
        raise AssertionError("Registered preview guard is absent or ambiguous")
    guard = compile(ast.Module(body=calls,type_ignores=[]), "registered-preview-guard", "exec")
    from types import SimpleNamespace
    scope = {"resume":resume,"model":SimpleNamespace(pipe=SimpleNamespace(scheduler=scheduler)),"scheduler":expected}
    exec(guard, scope)
    scheduler.timesteps[0] += 1
    try:
        exec(guard, scope)
    except (RuntimeError, ValueError) as error:
        if "scheduler state or implementation mismatch" not in str(error):
            raise
    else:
        raise AssertionError("A changed scheduler tensor was accepted")
    print(json.dumps({"native_timestep_elements":scheduler.timesteps.numel(),
                      "old_tensor_boolean_error_reproduced":True,
                      "fixed_unchanged_scheduler_accepted":True,
                      "changed_timestep_rejected":True,
                      "original_scheduler_SHA256":SOURCE_SHA256,"device_cards":0},sort_keys=True))


if __name__ == "__main__":
    main()
