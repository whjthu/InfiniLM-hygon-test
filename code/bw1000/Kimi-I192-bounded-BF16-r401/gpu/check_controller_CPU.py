"""Actual controller with recording Slurm/Popen and real temp artifacts, 0GPU."""
from pathlib import Path
import importlib.util
import json
import tempfile
import types


def main():
    d = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('actual401controller', d / 'controller.py')
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    calls = dict(account=0, starts=0, scancel=0, probes=0)
    proc = types.SimpleNamespace(pid=424242, poll=lambda: 0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / 'results').mkdir(); (root / 'logs').mkdir()
        step = '910371.999999'
        a = dict(completed=True, passed=True, PID=10001, normal_Runtime_close=True,
                 Torch_after_Runtime_close=True, GPU_count=1, Model_loaded=False, Runtime_world=1,
                 CCL=False, completed_calls=68, CAPI_returned=68, qualification_passed_calls=68, attempted=68, all_checkpoint_banks_readonly=True,
                 trials=[dict(passed=True, finite=True, readonly=True, output_tail_canary=True)] * 68)
        (root / 'results' / ('Kimi401-' + step + '.json')).write_text(json.dumps(a))
        c.verify = lambda r, g: (dict(CPU_receipt_SHA256='c' * 64, library_SHA256='d' * 64, CPU_source_manifest_SHA256='e' * 64), {})
        def start(r, pin):
            calls['starts'] += 1
            runtime = root / 'results/Kimi401-GPU-runtime.json'; q = json.loads(runtime.read_bytes())
            q['device_identity'] = dict(PCI_bus_ID='0000:01:00.0')
            runtime.write_text(json.dumps(q))
            return proc, ['recording-srun']
        c.start = start
        c.time.sleep = lambda s: None
        def check_output(args, **kw):
            if args[:4] == ['scontrol', 'show', 'job', '910371']:
                return 'UserId=aclsylqidf(1000) JobState=RUNNING NodeList=a02r3n16'
            if args[:3] == ['scontrol', 'show', 'hostnames']: return 'a02r3n16\n'
            if args[0] == 'sacct':
                calls['account'] += 1
                if calls['account'] <= 2: return ''
                if calls['account'] == 3: raise c.subprocess.TimeoutExpired(args, 30)
                return step + '|' + c.LABEL + '|COMPLETED|0:0|a02r3n16\n'
            raise AssertionError(args)
        def run(args, **kw):
            if args[0] == 'scancel':
                calls['scancel'] += 1; raise AssertionError('Healthy step canceled')
            assert args[0] == 'srun' and '--gres=none' in args
            calls['probes'] += 1
            return types.SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(dict(node='a02r3n16', passed=True,
                KFD_pids=[], devices=[dict(PCI_bus_ID='0000:01:00.0')])))
        c.subprocess.check_output = check_output; c.subprocess.run = run
        q = c.run(root, dict(execute_once=True, publication_commit='f' * 40, GPU_source_manifest_SHA256='a' * 64))
        assert q['passed'] and q['normal1'] and q['actual_step'] == step
        assert len(q['submissions']) == len(q['observation_errors']) == 1
        assert calls == dict(account=4, starts=1, scancel=0, probes=2)
        assert q['device_identity']['PCI_bus_ID'] == '0000:01:00.0'
    command = c.gpu_command('/recording-root')
    assert '--jobid=910371' in command and '--mem=32000M' in command and '--gres=dcu:1' in command
    return dict(completed=True, passed=True, actual_Controller_run=True, transient_account_samehandle=True,
                start1=True, cancel0=True, PCI_inventory_match=True, normal_post=True,
                GPU=False, Slurm=False, Runtime=False, leaves='recording only', original_deadline='worker540/shell570/Slurm600/controller900')


if __name__ == '__main__': print(json.dumps(main()))
