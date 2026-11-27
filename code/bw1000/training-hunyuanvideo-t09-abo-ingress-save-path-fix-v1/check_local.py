"""Run the exact frozen domain wrapper and common CLI with a no-device endpoint."""
from contextlib import redirect_stderr
from difflib import unified_diff
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = REPO / "local-only/hunyuan-abo-n14-fd-mount-carrier-v2/OPERATIVE_V2"
PUBLISHED = REPO / "code/bw1000"
CLI = PUBLISHED / "training-qwen-target-unified-framework-a68952c-v1/deltas/tools/hygon_training_recipe.py"
SIBLING = PUBLISHED / "training-qwen-native-update-contract-v1/deltas/tools/hygon_vlm_recipe.py"
MANIFEST_SHA = "84f66a9cc76b09ea98b9b5f4230a7984966e8b2f35e4338daa71a490e541cfc0"
OLD_SHA = "6d8c7f48ef7a652547a749396673fe1de98bad51e7de24498b536e7080777a2b"
NEW_SHA = "5ad2a1b352f849c8aa0bdfaf1c83444549faf536369ed0cc7736a982565a697a"
INGRESS_SHA = "e5cc4c868126875f0686f43ab7fdf0415d2cb60b6bebdd59c47c8ac4b42607ea"
SIBLING_SHA = "71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2"
CLI_SHA = "b218b9a6c1b5b80aaf6df5a73006b06c1529a30a77b1e4ed7b6a8a1d7a9b4f96"


def pinned(path, expected):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected, path.name
    return raw


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FirstForwardError(RuntimeError):
    pass


def exercise(root, contract_path, mode):
    # Copy only pinned Python sources into this disposable interpreter fixture.
    manifest = json.loads(pinned(ORIGINAL / "SOURCE_FILES.json", MANIFEST_SHA))["files"]
    source = root / "src"
    source.mkdir()
    for name in ("model_mount.py", "placement.py", "PLACEMENT.json"):
        (source / name).write_bytes(pinned(ORIGINAL / name, manifest[name]["SHA256"]))
    cli = source / "hygon_training_recipe.py"
    cli.write_bytes(pinned(CLI, CLI_SHA))
    (source / "hygon_vlm_recipe.py").write_bytes(pinned(SIBLING, SIBLING_SHA))
    launch_path = root / "ORIGINAL_LAUNCH.json"
    launch_path.write_bytes(pinned(ORIGINAL / "ORIGINAL_LAUNCH.json", manifest["ORIGINAL_LAUNCH.json"]["SHA256"]))
    pinned(ORIGINAL / "domain_ingress.py", INGRESS_SHA)
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(source))
    try:
        contract = load("contract", contract_path)
        # A Windows temp file can expose a different NTFS stamp through
        # lstat and fstat. This fixture covers the writer, not the Linux
        # input admission gate. Source and cache SHA checks still execute.
        if os.name == "nt":
            contract.stamp = lambda info: [info.st_size]
        ingress = load("_exact_frozen_ingress", ORIGINAL / "domain_ingress.py")
        return exercise_entry(root, cli, launch_path, contract, ingress, mode)
    finally:
        for name in ("contract", "placement", "model_mount", "hygon_vlm_recipe", "_exact_frozen_ingress"):
            sys.modules.pop(name, None)
        sys.path[:] = old_sys_path


def exercise_entry(root, cli, launch_path, contract, ingress, mode):
    assert ingress.save is contract.save and ingress.load is contract.load
    # Keep RUN as PurePosixPath, which is the .4543 failure site. Narrowly map
    # the Linux cache path to disposable fixture data. All record I/O is real.
    ingress.RUN = PurePosixPath(root.as_posix())
    cache_raw = b"offline synthetic conditioning, never read by a model"
    cache_path = root / "fixture-cache.bin"
    cache_path.write_bytes(cache_raw)
    ingress.CACHE_BYTES = len(cache_raw)
    ingress.CACHE_SHA = sha256(cache_raw).hexdigest()
    original_read = ingress.read
    def read(path, limit):
        return original_read(cache_path if path == ingress.CACHE else path, limit)
    ingress.read = read
    original_launch = json.loads(launch_path.read_bytes())
    mapping = {original_launch["config"]["training_cache"]: str(ingress.CACHE),
               original_launch["source"]: str(ingress.NATIVE_ALIAS), original_launch["weights"]: str(ingress.MODEL)}
    asset_path = root / "assets.json"
    asset_path.write_text(json.dumps(mapping), encoding="utf8")
    ctx = {"request": contract.REQUEST, "epoch": 1795777000, "step": "1103606.4543"}
    argv = [str(cli), "T09.sft", "--a100-launch", str(launch_path), "--asset-map", str(asset_path),
            "--cards", "0", "--output-root", "/fixture/output", "--run-name", "fixture", "--lock-root",
            "/fixture/lock", "--lock-host", "fixture", "--source-root", str(ingress.NATIVE_ALIAS.parent), "--run"]
    state = {"hook": None, "removed": 0, "registered": 0, "forward_entered": 0, "fsync": 0}

    class Muon:
        param_groups = [{"lr": 1e-5}, {"lr": 2e-6}]

    backend = root / "fixture-native-backend.py"
    program = compile("def train_step(optimizer, global_step, callback):\n    callback(optimizer, (), {})\n",
                      str(backend), "exec")
    synthetic = {}
    exec(program, synthetic)

    def parse_args(arguments):
        assert arguments[:3] == ["train", "--model", "hunyuanvideo-1.5"]
        assert arguments[arguments.index("--accelerator")+1] == "hip"
        recipe = json.loads(Path(arguments[arguments.index("--config")+1]).read_bytes())
        expected = dict(original_launch["config"], training_cache=str(ingress.CACHE), training_cache_sha=ingress.CACHE_SHA)
        assert recipe == expected
        return SimpleNamespace(handler=handler)

    sentinel = FirstForwardError("fixture first-forward error")
    def handler(unused):
        state["forward_entered"] += 1
        if mode == "forward-failure":
            raise sentinel
        optimizer = Muon()
        for step in range(1, 26):
            synthetic["train_step"](optimizer, step, state["hook"])
        return 0

    def register(hook):
        assert state["hook"] is None
        state["hook"] = hook
        state["registered"] += 1
        def remove():
            assert state["hook"] is hook
            state["removed"] += 1
            state["hook"] = None
        return SimpleNamespace(remove=remove)

    torch = ModuleType("torch")
    optimizer_module = ModuleType("torch.optim.optimizer")
    optimizer_module.register_optimizer_step_post_hook = register
    gencuda = ModuleType("gencuda")
    gencuda.cli = SimpleNamespace(parse_cards=lambda arg: [int(arg)],
                                 build_parser=lambda: SimpleNamespace(parse_args=parse_args))
    fsync = os.fsync
    def counted_fsync(fd):
        fsync(fd)
        state["fsync"] += 1
    def invoke():
        return ingress.run_common(torch, argv, ctx, cli, backend)
    with patch.dict(sys.modules, {"torch": torch, "torch.optim.optimizer": optimizer_module, "gencuda": gencuda}), \
         patch.object(os, "fsync", counted_fsync):
        if mode == "original":
            try:
                invoke()
            except AttributeError as error:
                assert "PurePosixPath" in str(error) and "open" in str(error)
                assert isinstance(error.__context__, AttributeError)
            else:
                raise AssertionError("frozen original did not reproduce the two save failures")
            assert state == dict(hook=None, removed=1, registered=1, forward_entered=0, fsync=0)
            assert ingress.progress(root, ctx)["evidence"] == "MISSING"
            return {"original_ingress_and_close_failure_reproduced": True}

        if mode == "forward-failure":
            try:
                invoke()
            except FirstForwardError as error:
                assert error is sentinel
            else:
                raise AssertionError("first-forward exception did not escape")
            count = 0
        else:
            assert invoke() == 0
            count = 25
        assert state == dict(hook=None, removed=1, registered=1, forward_entered=1, fsync=2)
        assert ingress.progress(root, ctx)["observed"] == count
        assert ingress.progress(root, ctx)["missing"] == 25-count
        ingress_path, muon_path = [root / name for name in ("domain-cache-ingress.json", "domain-muon.json")]
        pins = (ingress_path.read_bytes(), muon_path.read_bytes())
        meta, muon = map(json.loads, pins)
        assert meta["plan_calls"] == 1 and meta["cache"]["SHA256"] == ingress.CACHE_SHA
        assert muon["observed"] == count and muon["common_return"] == (None if count == 0 else 0)
        if mode == "complete":
            assert ingress.review(root, ctx)["actual_positive_LR_Muon_steps"] == 25
            with redirect_stderr(io.StringIO()):
                try:
                    invoke()
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("duplicate call must refuse the prior close record")
            assert (ingress_path.read_bytes(), muon_path.read_bytes()) == pins
            assert state == dict(hook=None, removed=2, registered=2, forward_entered=1, fsync=2)
        return dict(observed_fixture_hooks=count, missing=25-count, first_error_preserved=(count == 0))


def main():
    old = pinned(ORIGINAL / "contract.py", OLD_SHA)
    new = pinned(HERE / "contract.py", NEW_SHA)
    anchor = b'def save(path, value):\n    with path.open("x", encoding="utf-8") as stream:'
    assert old.count(anchor) == 1 and new == old.replace(anchor, anchor.replace(b"path.open(", b"Path(path).open("), 1)
    review_diff = "".join(unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
                                       fromfile="a/contract.py", tofile="b/contract.py", n=0)).encode()
    assert (HERE / "contract.patch").read_bytes() == review_diff
    with tempfile.TemporaryDirectory() as folder:
        results = {}
        for name in ("original", "complete", "forward-failure"):
            target = Path(folder) / name
            target.mkdir()
            results[name] = exercise(target, ORIGINAL / "contract.py" if name == "original" else HERE / "contract.py", name)
    print(json.dumps(dict(results=results, common_CLI=CLI_SHA, domain_ingress=INGRESS_SHA, corrected_contract=NEW_SHA,
                          actual_native_updates=0, actual_GPU=0), sort_keys=True))


if __name__ == "__main__":
    main()
