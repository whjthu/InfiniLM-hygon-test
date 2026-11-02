"""Once-only launch of four independent CPU keepers, each owns one GPU step."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

D = pathlib.Path(__file__).resolve().parent
R = pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sys.path.insert(0, str(D))


def load(replica):
    os.environ['DEEP_R375_REPLICA'] = str(replica)
    spec = importlib.util.spec_from_file_location('deep375_controller_r' + str(replica), D / 'controller_375.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execution-go-set', type=pathlib.Path, required=True)
    a = p.parse_args()
    raw = a.execution_go_set.read_bytes()
    go = json.loads(raw)
    assert go['one_attempt'] is True and go['GPU_authorized'] is True and len(go['replicas']) == 4
    modules = []
    for replica, g in enumerate(go['replicas']):
        c = load(replica)
        c.verify_sources(g)
        env = c.clean_env()
        c.parent(910407, 'a14r2n20', env)
        c.parent(c.PARENT, c.NODE, env)
        assert not c.W.exists()
        assert g['publication_commit'] == go['publication_commit']
        modules.append(c)
    intent = R / 'results/Deep-paired-quality-r375-all-intent.json'
    record = R / 'results/Deep-paired-quality-r375-all-keeper-launch.json'
    q = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), CPU_keepers_only=True,
             GPU_submitted_by_launcher=False, publication_commit=go['publication_commit'],
             execution_go_set_SHA256=hashlib.sha256(raw).hexdigest(), no_retry_reload=True,
             submissions=[])
    with intent.open('x') as f:
        json.dump(q, f, indent=2); f.flush(); os.fsync(f.fileno())
    with record.open('x') as f:
        json.dump(q, f, indent=2)
    for replica, c in enumerate(modules):
        path = R / 'results' / (c.LABEL + '-published-go.json')
        with path.open('x') as f:
            json.dump(go['replicas'][replica], f, indent=2)
        argv = ['srun', '--jobid=910407', '--overlap', '--exact', '--immediate=30', '--cpu-bind=none',
                '-N1', '-n1', '-c1', '-w', 'a14r2n20', '--gres=none', '--mem=4096M', '-t', '02:10:00',
                '--job-name=' + c.LABEL + '-keeper', '--output=' + str(R / 'logs' / (c.LABEL + '-keeper-%J.log')),
                '--export=ALL,DEEP_R375_REPLICA=' + str(replica),
                'python3', str(D / 'controller_375.py'), '--execution-go', str(path)]
        keeper_env = c.clean_env()
        keeper_env['DEEP_R375_REPLICA'] = str(replica)
        with (R / 'logs' / (c.LABEL + '-keeper-client.log')).open('x') as log:
            proc = subprocess.Popen(argv, env=keeper_env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        q['submissions'].append(dict(replica=replica, GPU_parent=c.PARENT, GPU_node=c.NODE,
                                      CPU_keeper_client_PID=proc.pid, actual_CPU_step='PENDING_NOT_YET_OBSERVED',
                                      actual_GPU_step='NOT_SUBMITTED_BY_THIS_LAUNCHER', argv=argv))
        temp = record.with_suffix('.tmp')
        temp.write_text(json.dumps(q, indent=2) + '\n'); temp.replace(record)
    print(json.dumps(q))


if __name__ == '__main__':
    main()
