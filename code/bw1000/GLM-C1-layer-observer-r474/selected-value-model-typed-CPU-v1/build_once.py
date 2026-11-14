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
PREVIOUS = Path('/refactor/results/GLM474-ffn-window-observer-CPU-v1')
J = read(W/'intent.json')
LAYER = Path('/refactor/results/GLM474-selected-layer0-Native-CPU-v2')
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
    plan = PREVIOUS/'src/BUILD_PLAN.json'
    assert H(plan)=='94aa31496788e23b74c1b0e611ae96ee16baea3ed8d61eea98ce199c4b680c4f'
    P = read(plan)
    OLD = Path(P['candidate'])/'CPU-result.json'
    assert H(OLD)=='9978a4f59a70ebc220c82b71e5f9d5c138bc28f9fc88e81e63b4c7676f720556'
    old=read(OLD)
    assert old['passed'] and old['actual_CPU']=='910407.1119'
    parent=LAYER/'libGLM_selected_layer0.so'
    assert H(parent)=='f73190b8a683bfb3bbcfbcb074b5f62f716e189ef1b4307cf87b4886547d178e'
    assert H(LAYER/'build-result.json')=='3c0a268533bb19b323f394dad3756f29e1d555ed943603b9f0628dfa41da556c'
    layer=read(LAYER/'build-result.json')
    assert layer['passed'] and layer['step']=='1103606.137' and layer['library_SHA256']==H(parent)
    assert H(PREVIOUS/'build-result.json')=='d300246ef893c1c3714490a86fd3c8bcbd06802b401487fb4ffcc9e984b81d35'
    ffn_build=read(PREVIOUS/'build-result.json')
    unit=P['units'][0]
    assert unit['name']=='unit0' and unit['source_SHA256']=='548eec8568e51057a1a683eb58d698e661bba8bcdca8b4643c088c63f0bc8ba0'
    pins=old['fresh_dependencies_SHA256'][unit['name']]
    original=unit['source']
    assert H(original)==pins[original]==unit['source_SHA256']
    prior_object=old['compiled'][unit['name']]['object']
    assert H(prior_object)==old['compiled'][unit['name']]['object_SHA256']
    obj=W/'full_bridge_observer.o'
    name='full_bridge_observer.cpp'
    source=str(D/name)
    expected={path:pin for path,pin in pins.items() if path!=original}
    expected[source]=J['source_files'][name]
    prefix=unit['prefix']
    fresh=W/'bridge.before.d'
    command('fresh-M',prefix+['-M','-MF',str(fresh),'-MT',str(obj),source])
    actual=dependencies(fresh)
    if actual!=set(expected):
        receipt['dependency_mismatch']=dict(missing=sorted(set(expected)-actual),extra=sorted(actual-set(expected)))
        raise RuntimeError('fresh dependency paths changed')
    for name,pin in expected.items():
        remaining()
        assert H(name)==pin,name
    receipt['fresh_M']=dict(count=len(actual),SHA256=H(fresh))
    compiled=W/'bridge.actual.d'
    command('compile',prefix+['-MD','-MF',str(compiled),'-MT',str(obj),'-c',source,'-o',str(obj)],90)
    receipt['new_HOST_TUs']=1
    assert dependencies(compiled)==actual
    for name,pin in expected.items():
        remaining()
        assert H(name)==pin,name
    receipt['MD']=dict(count=len(actual),SHA256=H(compiled),equal_fresh_M=True)
    original_objects=dict(old['all32_linked_objects_SHA256'])
    ffn=read(PREVIOUS/'src/FFN.json')
    assert H(PREVIOUS/'src/FFN.json')=='e29b9b6024f73dc3197e61e07e10ea9a37713d62d313a7ea026e11156d1f4c5c'
    original_objects.pop(ffn['original_object'])
    ffn_object=PREVIOUS/'routed_ffn_timed.o'
    assert H(ffn_object)==ffn_build['object_SHA256']
    original_objects[str(ffn_object)]=ffn_build['object_SHA256']
    assert len(original_objects)==32 and prior_object in original_objects
    swaps={old['compiled'][u['name']]['original_object']:old['compiled'][u['name']]['object'] for u in P['units']}
    swaps[ffn['original_object']]=str(ffn_object)
    old_nov='/refactor/candidates/GLM-global-TP16-full-native-r470-CPU-v4/NoV_tp16.o'
    original_objects.pop(old_nov)
    nov=LAYER/'NoV_tp16.o'
    assert H(nov)==layer['object_SHA256']
    original_objects[str(nov)]=layer['object_SHA256']
    swaps[old_nov]=str(nov)
    swaps[unit['original_object']]=str(obj)
    link=[swaps.get(arg,arg) for arg in P['link_argv']]
    candidate=W/'libGLM_selected_layer0_typed.so'
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
    assert {n:t for n,t in candidate_symbols.items() if n not in baseline_symbols}=={
        'modern_glm_global_selected_value_layer0_read':'T'}
    needed=elf(candidate,'candidate-elf')
    assert needed==elf(parent,'parent-elf')==old['NEEDED']
    dynamic=provider_hashes(candidate,'candidate-ldd')
    assert dynamic==old['new_transitive_library_SHA256'] and len(dynamic)==71
    receipt.update(passed=True, original_build_step=old['actual_CPU'], reused_objects=len(reused),
        linked_objects=32, object_SHA256=H(obj), library_SHA256=H(candidate), library_bytes=candidate.stat().st_size,
        strong_exports_preserved=len(baseline_symbols), diagnostic_exports=['modern_glm_global_selected_value_layer0_read'], providers=len(dynamic))

try:
    main()
except Exception as exc:
    receipt.update(error_type=type(exc).__name__, detail=str(exc)[:350])
finally:
    receipt['elapsed_seconds']=time.monotonic()-anchor
    with (W/'build-result.json').open('x') as output:
        json.dump(receipt,output,indent=2)
raise SystemExit(0 if receipt['passed'] else 1)
