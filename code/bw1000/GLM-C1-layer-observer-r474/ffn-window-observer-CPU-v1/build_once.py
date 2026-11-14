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
PARENT = Path(P['candidate'])
OLD = PARENT/'CPU-result.json'
F = read(D/'FFN.json')
ANCESTOR = Path('/refactor/candidates/GLM-global-TP16-full-native-r470-CPU-v4/CPU-result.json')
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

def symbols(path, label):
    out,_ = command(label, ['/usr/bin/nm','-D','--defined-only',str(path)])
    return {x[-1]:x[-2] for line in out.splitlines() if len(x:=line.split())>=3 and x[-2] in 'TDBR'}

def elf(path, label):
    out,_ = command(label,['/usr/bin/readelf','-d',str(path)])
    assert 'SYMBOLIC' in out
    return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',out)

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
    assert old['passed'] and old['actual_CPU']=='910407.1119' and old['all5_MD_equal_fresh_M']
    parent=Path(old['library'])
    assert H(parent)==old['library_SHA256']=='955e6a14060c636196a5afb1603f6b7405a63215dd5736f11c6f2f39adb7b18f'
    assert H(ANCESTOR)=='bb4db38050b127d92a74d82ae4cef83dd10da400a0e6c8b00c86030a71303c94'
    historical=read(ANCESTOR)
    assert historical['passed'] and historical['actual_CPU']=='910407.1087' and historical['all_fresh_M_equal_MD']
    name='routed_ffn_tp16.cpp'
    pins=historical['fresh_dependencies_SHA256'][name]
    prior=historical['compiled'][name]
    prior_object=F['original_object']
    assert prior['object']==prior_object and H(prior_object)==prior['object_SHA256']==F['original_object_SHA256']
    assert H(prior['MD'])==prior['MD_SHA256']=='4676ebca1091a6c4126d603c96358c38dcb5cb2de6376fac9d66cf6c2e21baae'
    assert dependencies(Path(prior['MD']))==set(pins) and len(pins)==993
    original=F['original_source']
    assert H(original)==pins[original]==prior['source_SHA256']=='02d0608d153e24776fbc3e0d0e827e1dafdcca7289a1698ef298128c254496f0'
    obj=W/'routed_ffn_timed.o'
    source=str(D/name)
    header=str(D/'glm_sparse_observer_r131.hpp')
    matches=[h for p,h in old['fresh_dependencies_SHA256'][U['name']].items()
             if p.endswith('/glm_sparse_observer_r131.hpp')]
    assert matches==[J['source_files']['glm_sparse_observer_r131.hpp']]
    expected={p:h for p,h in pins.items() if p!=original}
    expected.update({source:J['source_files'][name],header:J['source_files']['glm_sparse_observer_r131.hpp']})
    prefix=F['prefix']
    fresh=W/'routed_ffn.before.d'
    command('fresh-M',prefix+['-M','-MF',str(fresh),'-MT',str(obj),source])
    actual=dependencies(fresh)
    if actual!=set(expected):
        receipt['dependency_mismatch']=dict(missing=sorted(set(expected)-actual),extra=sorted(actual-set(expected)))
        raise RuntimeError('fresh dependency paths changed')
    for name,pin in expected.items():
        remaining()
        assert H(name)==pin,name
    receipt['fresh_M']=dict(count=len(actual),SHA256=H(fresh))
    compiled=W/'routed_ffn.actual.d'
    command('compile',prefix+['-MD','-MF',str(compiled),'-MT',str(obj),'-c',source,'-o',str(obj)],90)
    receipt['new_HOST_TUs']=1
    assert dependencies(compiled)==actual
    for name,pin in expected.items():
        remaining()
        assert H(name)==pin,name
    receipt['MD']=dict(count=len(actual),SHA256=H(compiled),equal_fresh_M=True)
    original_objects=old['all32_linked_objects_SHA256']
    assert len(original_objects)==32 and prior_object in original_objects
    swaps={old['compiled'][u['name']]['original_object']:old['compiled'][u['name']]['object'] for u in P['units']}
    swaps[prior_object]=str(obj)
    link=[swaps.get(arg,arg) for arg in P['link_argv']]
    candidate=W/'libGLM_ffn_window_observer.so'
    link[link.index('-o')+1]=str(candidate)
    objects=[arg for arg in link if arg.endswith('.o')]
    reused=set(original_objects)-{prior_object}
    assert len(objects)==len(set(objects))==32 and set(objects)==reused|{str(obj)}
    assert max(link.index(p) for p in objects)<min(i for i,arg in enumerate(link) if arg.startswith('-l'))
    for path in reused:
        assert H(path)==original_objects[path]
    command('link',link)
    baseline_symbols=symbols(parent,'parent-nm')
    candidate_symbols=symbols(candidate,'candidate-nm')
    assert all(candidate_symbols.get(name)==kind for name,kind in baseline_symbols.items())
    assert candidate_symbols==baseline_symbols
    needed=elf(candidate,'candidate-elf')
    assert needed==elf(parent,'parent-elf')==old['NEEDED']
    dynamic=provider_hashes(candidate,'candidate-ldd')
    assert dynamic==old['new_transitive_library_SHA256'] and len(dynamic)==71
    receipt.update(passed=True, original_build_step=old['actual_CPU'], reused_objects=len(reused),
        linked_objects=32, object_SHA256=H(obj), library_SHA256=H(candidate), library_bytes=candidate.stat().st_size,
        strong_exports_preserved=len(baseline_symbols), diagnostic_exports=[], providers=len(dynamic))

try:
    main()
except Exception as exc:
    receipt.update(error_type=type(exc).__name__, detail=str(exc)[:350])
finally:
    receipt['elapsed_seconds']=time.monotonic()-anchor
    with (W/'build-result.json').open('x') as output:
        json.dump(receipt,output,indent=2)
raise SystemExit(0 if receipt['passed'] else 1)
