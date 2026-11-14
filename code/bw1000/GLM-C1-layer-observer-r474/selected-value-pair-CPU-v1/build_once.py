"""Compile one GLM HOST unit against the qualified CPU1119 closure."""
from pathlib import Path
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import time

D = Path(__file__).resolve().parent
W = D.parent
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
P = read(D/'BUILD_PLAN.json')
J = read(W/'intent.json')
U = P['units'][4]
CANDIDATE = '/refactor/results/GLM474-ffn-window-observer-CPU-v1/libGLM_ffn_window_observer.so'
CANDIDATE_SHA = '017e981ace530ec762bf56bc981a73fd8bcb5c00afc4f5e8062403ac652a424e'
PARENT = Path(P['candidate'])
OLD = PARENT/'CPU-result.json'
EXPORTS = {'glm_tp16_selected_pair_create','glm_tp16_selected_pair_measure','glm_tp16_selected_pair_destroy'}
anchor = time.monotonic()
budget = J['work_deadline'] - time.time()
step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
receipt = dict(step=step, node=socket.gethostname().split('.')[0], passed=False,
               new_HOST_TUs=0, new_GPU_TUs=0, GPU_cards=0, commands=[])

def remaining():
    seconds = min(budget-(time.monotonic()-anchor), J['work_deadline']-time.time())
    if seconds <= 2:
        raise TimeoutError('original CPU build deadline')
    return seconds

def command(label, argv, limit=50):
    run = subprocess.run(argv, capture_output=True, text=True, timeout=min(limit, remaining()-1))
    for stream in ('stdout','stderr'):
        (W/f'{label}.{stream}').write_text(getattr(run, stream), encoding='utf-8')
    receipt['commands'].append(dict(name=label, rc=run.returncode,
       stdout_SHA256=H(W/f'{label}.stdout'), stderr_SHA256=H(W/f'{label}.stderr')))
    if run.returncode:
        raise RuntimeError(label+' returned '+str(run.returncode))
    return run.stdout, run.stderr

def dependencies(path):
    text = path.read_text().replace('\\\n',' ')
    _, separator, body = text.partition(':')
    names = set(shlex.split(body))
    assert separator and names and all(n.startswith('/') for n in names)
    return names

def provider_hashes(path, label):
    out,err = command(label,['/usr/bin/ldd','-r',str(path)],30)
    text=out+'\n'+err
    assert not any(x in text for x in ('not found','undefined symbol'))
    paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
    return {p:H(p) for p in sorted(paths)}

def main():
    assert receipt['node']==J['node']=='f13r1n05' and step.startswith('1103606.')
    assert all(os.environ.get(k)=='-1' for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))
    assert not os.environ.get('LD_PRELOAD')
    assert all(H(D/name)==pin for name,pin in J['source_files'].items())
    tools=read(D/'TOOLCHAIN.json')
    for path,pin in tools['tools'].items():
        assert str(Path(path).resolve())==pin['canonical'] and H(path)==pin['sha256']
    for key,value in tools['required_environment'].items():
        assert os.environ.get(key)==value
    assert H(OLD)=='9978a4f59a70ebc220c82b71e5f9d5c138bc28f9fc88e81e63b4c7676f720556'
    old=read(OLD)
    pins=old['fresh_dependencies_SHA256'][U['name']]
    assert len(pins)==992 and old['passed'] and old['actual_CPU']=='910407.1119'
    candidate=Path(CANDIDATE)
    assert H(candidate)==CANDIDATE_SHA
    source=D/'selected_pair.cpp'
    obj=W/'selected_pair.o'
    prefix=U['prefix']+['-I'+str(D)]
    fresh=W/'selected_pair.before.d'
    command('fresh-M',prefix+['-M','-MF',str(fresh),'-MT',str(obj),str(source)])
    names=dependencies(fresh)
    # The external caller's canonical include spelling is already present in
    # the original plan's mirror, even when unit4 used an aliased spelling.
    relative='candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/weights/weights.hpp'
    weights_header=str(PARENT/'project-mirror'/relative)
    weights_pin='1056d753a7c5be68ef493b547062eea63a48bb218d710b2616aeac60ca379cca'
    assert P['mirror_files'][relative]['effective_SHA256']==weights_pin
    assert any(Path(path).resolve()==Path(weights_header) and h==weights_pin for path,h in pins.items())
    pins={**pins, weights_header:weights_pin}
    # Accept only exact path and hash pairs from the qualified build mirror.
    for relative, row in P['mirror_files'].items():
        p = str(PARENT/'project-mirror'/relative)
        if p in names and p not in pins:
            pins[p] = row['effective_SHA256']
    assert H(candidate.parent/'build-result.json')=='d300246ef893c1c3714490a86fd3c8bcbd06802b401487fb4ffcc9e984b81d35'
    own={str(D/name):sha for name,sha in J['source_files'].items()
         if name.endswith(('.cpp','.hpp'))}
    assert set(own)<=names
    extra=names-set(pins)-set(own)
    assert not extra, ('new unregistered inputs', sorted(extra))
    expected={name:(own[name] if name in own else pins[name]) for name in names}
    assert set(expected)==names and any(p.endswith('/glm_moe_dsa_attention_prepare.hpp') for p in names)
    assert all(H(p)==h for p,h in expected.items())
    receipt['fresh_M']=dict(count=len(names),SHA256=H(fresh))
    md=W/'selected_pair.actual.d'
    command('compile',prefix+['-MD','-MF',str(md),'-MT',str(obj),'-c',str(source),'-o',str(obj)],90)
    receipt['new_HOST_TUs']=1
    assert dependencies(md)==names and all(H(p)==h for p,h in expected.items())
    receipt['MD']=dict(count=len(names),SHA256=H(md),equal_fresh_M=True)
    link=P['link_argv']
    tail=link[link.index('-L/refactor/candidates/glm-grouped-native-model-r5117/helpers'):link.index('-o')]
    assert '-Wl,-Bsymbolic' in tail
    tail=[x for x in tail if x!='-Wl,-Bsymbolic']
    binary=W/'libGLM_selected_value_pair.so'
    command('link',[prefix[0],'-shared',str(obj),str(candidate),'-Wl,-rpath,'+str(candidate.parent),*tail,
                    '-pthread','-Wl,--no-undefined','-o',str(binary)])
    mapped=provider_hashes(binary,'ldd')
    assert mapped.pop(str(candidate))==CANDIDATE_SHA
    assert mapped==old['new_transitive_library_SHA256'] and len(mapped)==71
    exported,_=command('exports',['/usr/bin/nm','-D','--defined-only',str(binary)])
    assert {line.split()[-1] for line in exported.splitlines() if line.split() and line.split()[-1].startswith('glm_tp16_')}==EXPORTS
    receipt.update(passed=True, library_SHA256=H(binary), library_bytes=binary.stat().st_size,
                   original_native_SHA256=CANDIDATE_SHA, object_SHA256=H(obj),
                   verified_exports=sorted(EXPORTS), providers=len(mapped),
                   component_executed=False,runtime_created=False)


try:
    main()
except Exception as exc:
    receipt.update(error_type=type(exc).__name__, detail=str(exc)[:350])
finally:
    receipt['elapsed_seconds']=time.monotonic()-anchor
    with (W/'build-result.json').open('x') as output:
        json.dump(receipt,output,indent=2)
raise SystemExit(0 if receipt['passed'] else 1)
