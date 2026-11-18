"""Only newly introduced 40GB/held-wrapper/zero-byte boundaries, no old fixtures."""
import ast
import json
from pathlib import Path
import gpu_admission as gpu

D = Path(__file__).resolve().parent


def main():
    results = []
    plan = json.loads((D / 'INSTALL_PLAN.json').read_bytes()); expected = plan['expected_GPU']
    saved = gpu.query, gpu.selected_fd_holders, gpu.socket.socket
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def bind(self, address): assert address == ('127.0.0.1', plan['master_port'])
    try:
        gpu.socket.socket = Socket
        gpu.selected_fd_holders = lambda *_: dict(checked_own_UID_processes=0, selected_device_or_KFD_holders=[], incomplete_processes=0)
        for name, total, free, wanted in [('actual40GB',40960,40442,True), ('wrong80GB',81920,80000,False), ('insufficient_free',40960,37999,False)]:
            gpu.query = lambda fields, kind, deadline: [[str(expected['index']), expected['UUID'], expected['PCI'], '0', str(free), str(total)]] if kind == 'gpu' else []
            try:
                gpu.snapshot(expected, plan['master_port'], 10**20); accepted = True
            except ValueError:
                accepted = False
            results.append(dict(name=name, passed=accepted == wanted))
    finally:
        gpu.query, gpu.selected_fd_holders, gpu.socket.socket = saved
    child = ast.parse((D / 'worker_entry.py').read_bytes())
    byte_splits = [n.args[0].value for n in ast.walk(child) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'split' and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, bytes)]
    results.append(dict(name='actual_child_cmdline_split_is_NUL_00', passed=byte_splits == [bytes([0])]))
    tree = ast.parse((D / 'qualified/bounded_GPU_supervisor.py').read_bytes())
    original = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP' for t in n.targets))
    old = 'os.execv(sys.executable,[sys.executable]+sys.argv[1:])'
    source = (D / 'bounded_eval.py').read_text(encoding='utf8')
    results.append(dict(name='actual_qualified_BOOTSTRAP_one_replacement', passed=original.count(old) == 1 and
        "need(q.BOOTSTRAP.count(old_exec) == 1" in source))
    from bounded_eval import fresh_before
    for name, before, now, wanted in [('samehost_fresh60', 100, 160, True), ('samehost_stale61', 100, 161, False)]:
        try:
            fresh_before(before, now); accepted = True
        except ValueError:
            accepted = False
        results.append(dict(name=name, passed=accepted == wanted))
    import submit_once
    main_node = next(n for n in ast.parse((D / 'submit_once.py').read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    results.append(dict(name='client_epoch_before_verify_and_Git', passed=isinstance(main_node.body[0], ast.Assign)
        and isinstance(main_node.body[0].targets[0], ast.Name) and main_node.body[0].targets[0].id == 'epoch'))
    original_run = submit_once.subprocess.run; observed = {}
    class Result:
        returncode = 0
        stdout = b'synthetic'
    def fake_run(*args, **kwargs):
        observed.update(kwargs); return Result()
    try:
        submit_once.subprocess.run = fake_run
        submit_once.git('show', 'synthetic', budget=lambda: 2.5)
        results.append(dict(name='actual_Git_timeout_shrinks_to_remaining', passed=observed['timeout'] == 2.5))
    finally:
        submit_once.subprocess.run = original_run
    value = dict(passed=all(r['passed'] for r in results), cases=results, synthetic_only=True,
                 HTTP=0, actual_nvidia_smi=0, actual_Torch=0, actual_remote=0, actual_GPU=0, old_tests_repeated=False)
    print(json.dumps(value, sort_keys=True)); return 0 if value['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
