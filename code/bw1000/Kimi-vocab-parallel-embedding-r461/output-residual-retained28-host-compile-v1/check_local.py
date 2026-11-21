"""Run real source rendering through the inherited host controller and recorded I/O."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "output-residual-wave-host-compile-v1/check_local.py"
FIXTURE_SHA256 = "a9a639a9f395b408c029fb2a352144e2365da8eb77712dae659a66e1d7df0339"


def run():
    original = FIXTURE.read_bytes()
    if sha256(original).hexdigest() != FIXTURE_SHA256:
        raise ValueError("frozen host fixture differs")
    text = original.decode()
    edits = (
        ('with_name("host_once.py"))))', 'with_name("host_once.py")))["controller"]())'),
        ('planned_device_TUs=2, actual_device_TUs=2,', 'planned_device_TUs=1, actual_device_TUs=1,'),
        ('for index, arm in enumerate(H.ARMS):', 'for arm in H.ARMS:'),
        ('case == "wrong_elf_type" and index else', 'case == "wrong_elf_type" else'),
        ('        elf[-1] = index\n', ''),
        ('case == "md_dependency_drift" and index else', 'case == "md_dependency_drift" else'),
        ('case == "wrong_symbol" and index else', 'case == "wrong_symbol" else'),
        ('(path / "full-tree" / "fused_scores.cu")', '(path / "retained28" / "fused_scores.cu")'),
        ('b"f12r2n11", b"f12r2n12"', 'b"f12r2n10", b"f12r2n12"'),
        ('exact_rendered_members=13', 'exact_rendered_members=7'),
        ('    if case == "command_drift":',
         '    if case == "wrong_planned_width":\n        receipt["planned_device_TUs"] = 2\n'
         '    if case == "command_drift":'),
        ('"command_drift", "unexpected_gres", "slurm_failed")',
         '"command_drift", "unexpected_gres", "slurm_failed", "wrong_planned_width")'),
    )
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError("frozen fixture anchor differs")
        text = text.replace(old, new)
    scope = dict(__name__="retained28_host_fixture", __file__=__file__)
    exec(compile(text, str(FIXTURE), "exec"), scope)
    host = scope["H"]
    with TemporaryDirectory() as tmp:
        target = Path(tmp).resolve() / "rendered"
        scope["RENDER"](target)
        manifest = host.generated_payload(target)
        plan = json.loads((target / "BUILD_PLAN.json").read_bytes())
        worker = ast.parse((target / "compile_once.py").read_bytes())
        declarations = [node.value.value for node in ast.walk(worker)
                        if isinstance(node, ast.keyword) and node.arg == "planned_device_TUs"
                        and isinstance(node.value, ast.Constant)]
        assert declarations == [1] and plan["planned_device_TUs"] == 1
        assert manifest["retained28/fused_scores.cu"] == (
            "bc530d46b82723ddb1279f1000af1fef10dba32003902a862ab474d4648db410")
        assert [name for name, _ in host.compile_commands(plan)] == [
            "retained28-M", "retained28-compile", "retained28-symbols"]
    scope["main"]()


if __name__ == "__main__":
    run()
