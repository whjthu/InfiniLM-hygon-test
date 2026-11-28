"""Check the production OneVision observer against the target Swift pre-hook order."""

import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace


def main():
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert os.environ[key] == "-1"
    assert not os.environ.get("SLURM_STEP_GPUS") and not os.environ.get("SLURM_GPUS_ON_NODE")
    sys.path[:0] = ["/tmp/patch", "/llava-helpers", "/llava-framework", "/swift-runtime/site", "/vision", "/llava-source"]
    import callback
    import torch
    from swift.template.base import Template

    source = Path(inspect.getfile(Template))
    assert source == Path("/swift-runtime/site/swift/template/base.py")
    assert sha256(source.read_bytes()).hexdigest() == "cc1629c8a84b764b530895692e4e04322aa6732281be229efce9e98752e113cc"
    production = Path(callback.__file__)
    assert production == Path("/tmp/patch/callback.py")
    raw = production.read_bytes()
    registrations = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "register_forward_pre_hook"]
    [registration] = registrations
    bound_call = compile(ast.Expression(registration), str(production), "eval")

    class Model(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *, inputs_embeds, labels, attention_mask):
            self.calls += 1
            return inputs_embeds.sum(), labels.sum(), attention_mask.sum()

    class SwiftFixture:
        pre_forward_hook = Template.pre_forward_hook
        register_post_encode_hook = Template.register_post_encode_hook

        def __init__(self):
            self._handles = []

        @staticmethod
        def get_base_model(model):
            return model

        @staticmethod
        def _post_encode(model, inputs):
            return {"inputs_embeds": inputs["input_ids"].to(torch.float32).unsqueeze(-1)}

    def observe(mode):
        inputs = {"input_ids": torch.tensor([[11, 23, 37]]), "labels": torch.tensor([[-100, 23, 37]]),
                  "attention_mask": torch.tensor([[1, 1, 1]]), "pixel_values": torch.ones((1, 3, 2, 2))}
        key = callback.input_sha(inputs)
        model = Model()
        swift = SwiftFixture()
        swift.register_post_encode_hook([model])
        events = []
        observer = SimpleNamespace(forwards=[], deadline=float("inf"), steps=0, prepared=key,
                                   prepare_shape=callback.geometry(inputs),
                                   bounds={key: {"ordinal": 1, "image_sha256": "a" * 64, "row_sha256": "b" * 64}},
                                   report=lambda name, **kw: events.append((name, kw)))
        observer.forward = lambda *a: callback.LlavaClevrFirstTwo.forward(observer, *a)
        if mode == "fixed":
            handle = eval(bound_call, {"model": model, "self": observer})
        elif mode == "late":
            handle = model.register_forward_pre_hook(observer.forward, with_kwargs=True)
        else:
            handle = None
        try:
            output = [value.item() for value in model(**inputs)]
        except KeyError as error:
            assert mode == "late" and error.args == ("input_ids",) and model.calls == 0
            assert len(events) == 1 and events[0][0] == "input-mismatch.json"
            return {"mode": mode, "reproduced_late_keyerror": True, "module_calls": model.calls}
        finally:
            if handle is not None:
                handle.remove()
            for _, swift_handle in swift._handles:
                swift_handle.remove()
        assert model.calls == 1 and output == [71.0, -40, 3]
        if mode == "fixed":
            [(event, fields)] = events
            assert event == "forward-01.json" and fields["input_sha256"] == key and observer.prepared is None
        return {"mode": mode, "module_calls": model.calls, "fixture_output": output}

    observations = [observe(mode) for mode in ("swift_only", "late", "fixed")]
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        assert target != "/dev/kfd" and not re.fullmatch(r"/dev/dri/(?:renderD|card)\d+", target)
    print(json.dumps(dict(case="SWIFT_NATIVE_PREHOOK_ORDER", records=observations,
                          callback_SHA256=sha256(raw).hexdigest(), training_updates=0, GPU_fds=0),
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
