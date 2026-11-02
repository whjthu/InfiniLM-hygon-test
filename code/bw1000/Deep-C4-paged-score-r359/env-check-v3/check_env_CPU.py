"""Reuse CPU714 objects; inspect linking and CPU reference without dlopen/GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
R=Path('/refactor')
S=R/'incoming/Deep-C4-paged-score-r359-source-cpu-v2'
B=R/'candidates/Deep-C4-paged-score-r359-source-cpu-v2'
D=R/'candidates/Deep-C4-paged-score-r359-env-check-v3'
D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,
       recompiled=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
start=time.monotonic()
try:
    assert sha(B/'CPU-build.json')=='f741ded4e1c9d4786d1a9c6108040efa0521953efa89e8072e65281776ea4640'
    old=json.loads((B/'CPU-build.json').read_bytes())
    assert old['completed'] is False
    assert sha(S/'SOURCE_MANIFEST.json')=='38f8a95cfa8c6c7669ea43f66d1cb7d9507182a167feae23e2e93e1c762fc4ae'
    m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
    lib=B/'libDeep_C4_private_source_r359.so'
    assert sha(lib)=='5612ab74c4f177753c7ecdcb292f55b25f9ca10d2c115e8b77e48c9c3384b4f3'
    pins={
      '/refactor/candidates/latest-family-native-r269/Deep/python/_infinilm.cpython-310-x86_64-linux-gnu.so':'6e1a7e7643f84a52ab147cfc3c55eee4816affc6898109dbef31d559feaa86d1',
      '/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so':'c04e19df5fa07bb52c0acf7be9768d258f5e2aa706647db32761e23426b153ec',
      '/refactor/install/hygon-deep-5f91ad7-r3672/lib/libinfiniops.so':'e1d9432e6de79cf09782fc3b3b332fb969d1a948dbac513b3da677d78f7f1f09',
      '/refactor/install/hygon-deep-5f91ad7-r3672/lib/libinfinirt.so':'e78b7b4aed105e5968fc598f60a2cacafcaf7c07c13dd4a971abfbdd92d47560',
      '/opt/dtk/cuda/cuda/lib64/libcudart.so.12':'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e',
      '/refactor/incoming/Deep-HC-epilogue-r358-cpu-r2/CPU_inside.sh':'501bbd86ebc6e6a21d4f153ecddb7f8146afd95eb704768064511e51d05be66a'}
    for p,h in pins.items():assert sha(p)==h,p
    q['dependency_pins_SHA256']=pins
    q['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']
    q['artifacts_before_SHA256']={n:sha(B/n) for n in ['private.o','score.o','private.d','libDeep_C4_private_source_r359.so']}
    def run(args,label):
        p=subprocess.run(args,capture_output=True,text=True,timeout=60)
        (D/(label+'.log')).write_text(p.stdout+p.stderr)
        assert p.returncode==0,(label,p.returncode,p.stderr[-2000:])
        return p.stdout
    ld=run(['ldd',str(lib)],'ldd')
    assert 'not found' not in ld,ld
    # Both private DSO and original Native must resolve under the same environment.
    native=R/'candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'
    assert sha(native)=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
    assert 'not found' not in run(['ldd',str(native)],'native-ldd')
    nm=run(['nm','-D','--defined-only',str(lib)],'nm')
    assert 'deep_paged_scores_r359' in nm and 'paged_score_r359' in nm
    ref=json.loads(run(['python','-B',str(S/'check_reference_CPU.py')],'reference'))
    assert ref['passed'] and ref['GPU']==0
    q['artifacts_after_SHA256']={n:sha(B/n) for n in q['artifacts_before_SHA256']}
    assert q['artifacts_before_SHA256']==q['artifacts_after_SHA256']
    q.update(completed=True,reference=ref,source_manifest_SHA256=sha(S/'SOURCE_MANIFEST.json'),
             ldd_SHA256=sha(D/'ldd.log'),native_ldd_SHA256=sha(D/'native-ldd.log'),
             nm_SHA256=sha(D/'nm.log'),old_CPU_failure_preserved=True,
             Runtime_adapter='PENDING',GPU_GO=False)
except BaseException as e:
    q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start
with (D/'CPU-env-check.json').open('x') as f:json.dump(q,f,indent=2)
print(json.dumps(q,indent=2))
assert q['completed']
