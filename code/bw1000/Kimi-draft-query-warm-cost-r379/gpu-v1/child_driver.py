"""One fresh Python process, original strict draft loader and original r333 query."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import pathlib
import socket
import sys
import time
import traceback

from warm_plan import run_contexts

D = pathlib.Path(__file__).resolve().parent
R = pathlib.Path('/refactor')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    entered = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--deadline-unix', required=True, type=float)
    parser.add_argument('--index', required=True, type=int, choices=range(4))
    parser.add_argument('--warm', required=True, type=int, choices=[0, 1])
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    assert os.environ.get('KIMI_WARM_COST_R379') == '1'
    assert [0, 1, 1, 0][args.index] == args.warm
    assert 20 < args.deadline_unix-time.time() <= 600
    assert sha(D/'GPU_MANIFEST.json') == args.manifest_sha
    manifest = json.loads((D/'GPU_MANIFEST.json').read_bytes())
    assert manifest['GPU_GO'] is True
    assert all(sha(D/name) == expected for name, expected in manifest['files_SHA256'].items())
    assert all(sha(path) == expected for path, expected in manifest['dependencies_SHA256'].items())
    assert not args.output.exists()
    report = dict(completed=False, passed=False, actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                  child_index=args.index, warm_query=bool(args.warm), PID=os.getpid(), node=socket.gethostname(),
                  source_manifest_SHA256=args.manifest_sha, target_cards=0, draft_cards=1,
                  profile=False, no_cache_or_crop=True, physical_GPU_cold='UNKNOWN',
                  scope='Fresh process first measured original draft query; synthetic features, not target/online throughput')

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)

    def remaining():
        if time.time() >= args.deadline_unix-10:
            raise TimeoutError('Original whole-step deadline, no process retry')

    runtime = None
    started = entered
    save()
    try:
        old = R/'incoming/Kimi-long-draft-memory-floor-r362b'
        sys.path.insert(0, str(old))
        import worker
        assert pathlib.Path(worker.__file__).resolve() == old/'worker.py'
        runtime_directory, incoming = worker.verify_sources(old, R, manifest['worker_manifest_SHA256'])
        remaining()
        import torch
        torch.set_num_threads(1)
        with torch.inference_mode():
            runtime = worker.Runtime(R, runtime_directory, incoming, report)
            assert runtime.loaded.worker.model.training is False
            remaining()
            properties = torch.cuda.get_device_properties(0)
            assert 'gfx936' in properties.gcnArchName
            cuda = ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so')
            cuda.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            bus = ctypes.create_string_buffer(64)
            assert cuda.cudaDeviceGetPCIBusId(bus, 64, 0) == 0
            report['PCI'] = bus.value.decode()
            report['runtime_class'] = type(runtime.session.context).__module__+'.'+type(runtime.session.context).__qualname__
            assert report['runtime_class'] == 'long_context_r323.LongCommittedFeatureContext'
            report['feature_fixture'] = {}
            feature_bytes = {}
            for role, seed in [('priming', 379), ('measured', 324)]:
                generator = torch.Generator(device='cpu').manual_seed(seed)
                block = (torch.randn(1, 512, 35840, dtype=torch.bfloat16, generator=generator)*.1).contiguous()
                data = block.view(torch.uint16).numpy().tobytes()
                feature_bytes[role] = data
                report['feature_fixture'][role] = dict(seed=seed, rows=512, raw_dim=35840,
                                                       SHA256=hashlib.sha256(data).hexdigest())
            assert feature_bytes['priming'] != feature_bytes['measured']
            report['strict_load_and_fixture_setup_s'] = time.perf_counter()-started
            torch.cuda.reset_peak_memory_stats(0)

            def save_plan(value):
                report['query_plan'] = value
                save()

            plan = run_contexts(runtime.session, bool(args.warm),
                                lambda role, rows: feature_bytes[role][:rows*35840*2],
                                save=save_plan, remaining=remaining)
            report['query_plan'] = plan
            report['normal_session_close'] = runtime.session.closed
            report['Torch_peak_allocated_after_load_bytes'] = torch.cuda.max_memory_allocated(0)
            report['Torch_peak_reserved_after_load_bytes'] = torch.cuda.max_memory_reserved(0)
            report['memory_scope'] = 'Torch allocator only, excludes non-Torch HIP allocations; original per-command4GiB floor unchanged'
            report['passed'] = True
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
                report['normal_Runtime_close'] = True
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            gc.collect()
            report.update(completed=True, whole_child_wall_s=time.perf_counter()-started)
            save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
