"""Actual staged package/defaultD/main asset preflight with real original fixture bytes.

Only the /refactor raw-file root is remapped to the local evidence directory.
Device/weight leaves are replaced by a refusing boundary, not claimed as qualified.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types


def check(raw_directory):
    raw_directory = Path(raw_directory).resolve()
    d = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp).resolve() / 'actual-staged-gpu-v2'; stage.mkdir()
        for name in ['GPU_run.py', 'plan.py', 'FIXTURE_PINS.json']:
            shutil.copyfile(d / name, stage / name)
        spec = importlib.util.spec_from_file_location('actual401assetmain', stage / 'GPU_run.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        assert module.D == stage
        seen = []
        def resolve(p):
            seen.append(p); return raw_directory / Path(p).name
        arrays = module.load_prepared_assets(resolve)
        assert len(seen) == 3 and [list(v.shape) for v in arrays] == [[16,3584],[16,16],[16,16]]
        pins = json.loads((stage / 'FIXTURE_PINS.json').read_bytes())
        assert all(hashlib.sha256((raw_directory / Path(p).name).read_bytes()).hexdigest() == pins[Path(p).name]['SHA256'] for p in seen)
        checks = ['actual_staged_directory_defaultD_metadata', 'three_actual_raw_bytes_and_hashes', 'validated_small_arrays_cached']
        class BeforeDeviceBoundary(RuntimeError): pass
        calls = []
        torch = types.ModuleType('torch')
        def refuse(n):
            assert len(seen) == 6
            calls.append('torch-boundary'); raise BeforeDeviceBoundary('No actual Torch/device/weights in CPU test')
        torch.set_num_threads = refuse
        safe = types.ModuleType('safetensors')
        safe.safe_open = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Weight opened in CPU check'))
        old = {n: sys.modules.get(n) for n in ['torch','safetensors']}
        sys.modules.update(torch=torch, safetensors=safe)
        try:
            try: module.main('must-not-read-manifest', 'must-not-create-report', asset_resolver=resolve)
            except BeforeDeviceBoundary: pass
            else: raise AssertionError('Actual main did not reach the intended refusing boundary')
            assert calls == ['torch-boundary']
            checks.append('actual_main_reads_all_assets_before_Torch_device_weights')
            (stage / 'FIXTURE_PINS.json').unlink()
            seen.clear(); calls.clear()
            try: module.main('must-not-read-manifest', 'must-not-create-report', asset_resolver=resolve)
            except FileNotFoundError as e: assert e.filename == str(stage / 'FIXTURE_PINS.json')
            else: raise AssertionError('Missing metadata admitted')
            assert not calls and not seen
            checks.append('missing_metadata_refused_before_Torch_device_weights')
        finally:
            for n, value in old.items():
                if value is None: sys.modules.pop(n, None)
                else: sys.modules[n] = value
    return dict(completed=True, passed=True, checks=checks, Runtime=False, GPU=False, Slurm=False,
                actual_main_asset_boundary=True, raw_bytes='Original pinned teacher16 threefiles, local-only; not synthetic replacements',
                leaves='Only actual device/weights boundary refuses, no numerical/device qualification')


if __name__ == '__main__': print(json.dumps(check(sys.argv[1])))
