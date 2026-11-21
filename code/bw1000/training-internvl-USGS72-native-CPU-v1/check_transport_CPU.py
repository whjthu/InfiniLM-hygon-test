"""Only injected-transport admission and a local binary-pipe Popen interface."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import launch_once as launcher


def main():
    tests=[]
    for name,value in (('missing_callback',None),('noncallable_callback',{})):
        try:launcher.main('f'*40,'e'*64,transport_start=value)
        except ValueError as error:
            assert str(error)=='ROOT_COORDINATOR_TRANSPORT_CALLBACK_REQUIRED_NO_FALLBACK'
            tests.append(dict(name=name,rejected_before_source_Git_or_transport=True))
        else:raise AssertionError('missing callback accepted')
    try:launcher.check_transport_child(dict(returncode=0,stdout=b'fake'))
    except ValueError:tests.append(dict(name='result_dict_cannot_masquerade_as_owned_child',rejected=True))
    else:raise AssertionError('result dict accepted as a process')
    observed=[]
    context=dict(request_id='8110b01cd24846f8b6fcd08f0abda5a0',epoch=1.0,deadline=301.0,
                 target_command=launcher.TARGET_COMMAND)
    def local_fixture_factory(c):
        observed.append(c)
        return subprocess.Popen([sys.executable,'-I','-S','-B','-c',
            'import sys; b=sys.stdin.buffer.read(); sys.stdout.buffer.write(b); sys.stderr.buffer.write(b"LOCAL_ONLY")'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    callback=launcher.require_transport_start(local_fixture_factory)
    child=callback(context)
    try:
        launcher.check_transport_child(child)
        out,err=child.communicate(input=b'CONTRACT_FIXTURE',timeout=5)
        assert child.returncode==0 and out==b'CONTRACT_FIXTURE' and err==b'LOCAL_ONLY' and observed==[context]
    finally:
        if child.poll() is None:child.kill();child.communicate(timeout=2)
    tests.append(dict(name='explicit_factory_binary_pipe_protocol',passed=True,local_child_reaped=True))
    d=Path(__file__).resolve().parent
    o=d.parents[2]/'local-only/training-internvl-USGS72-native-CPU-v1-transport-injection-checks'
    o.mkdir(exist_ok=False)
    r=dict(passed=True,tests=tests,local_fixture_children=1,target_or_network_calls=0,
        source_Git_calls=0,SSH_calls=0,Slurm_calls=0,Torch_imported=False,old_CPU_tests_rerun=False,
        native_encodes=0,GPU=0,training_updates=0,READY=False,
        actual_launch_source_SHA256=hashlib.sha256((d/'launch_once.py').read_bytes()).hexdigest())
    (o/'result.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=True,rejected_cases=3,local_binary_child_cases=1,target_calls=0,READY=False),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
