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
CANDIDATE = '/refactor/results/GLM474-projection-host-timing-CPU-v3/libGLM_projection_host_timing.so'
CANDIDATE_SHA = 'ad4186d84766a034e7ef4085539cf7c338454e68d629586649cbde235bbede77'
PARENT = Path(P['candidate'])
OLD = PARENT/'CPU-result.json'
EXPORTS = {'modern_glm_projection_host_timing_begin','modern_glm_projection_host_timing_end'}
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
    source=D/'projection_component.cpp'
    obj=W/'projection_component.o'
    prefix=U['prefix']
    fresh=W/'projection_component.before.d'
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
    extra=names-set(pins)-{str(source)}
    assert not extra, ('new unregistered inputs', sorted(extra))
    expected={name:pins[name] for name in names if name!=str(source)}
    expected[str(source)]=J['source_files'][source.name]
    assert set(expected)==names and any(p.endswith('/glm_moe_dsa_attention_prepare.hpp') for p in names)
    assert all(H(p)==h for p,h in expected.items())
    receipt['fresh_M']=dict(count=len(names),SHA256=H(fresh))
    md=W/'projection_component.actual.d'
    command('compile',prefix+['-MD','-MF',str(md),'-MT',str(obj),'-c',str(source),'-o',str(obj)],90)
    receipt['new_HOST_TUs']=1
    assert dependencies(md)==names and all(H(p)==h for p,h in expected.items())
    receipt['MD']=dict(count=len(names),SHA256=H(md),equal_fresh_M=True)
    link=P['link_argv']
    tail=link[link.index('-L/refactor/candidates/glm-grouped-native-model-r5117/helpers'):link.index('-o')]
    assert '-Wl,-Bsymbolic' in tail
    tail=[x for x in tail if x!='-Wl,-Bsymbolic']
    binaries={}
    for arm,lib,lib_sha in (
        ('baseline',candidate,CANDIDATE_SHA),
        ('single-drain',Path('/refactor/results/GLM474-projection-single-drain-CPU-v1/libGLM_projection_host_timing.so'),
         'abdf704dcf37898b3477ec2df2b1af6f8c456cb534edfa24d110666e17669b07'),
    ):
        assert H(lib)==lib_sha
        binary=W/('projection-'+arm)
        command(arm+'-link',[prefix[0],str(obj),str(lib),'-Wl,-rpath,'+str(lib.parent),*tail,
                            '-pthread','-Wl,--no-undefined','-o',str(binary)])
        mapped=provider_hashes(binary,arm+'-ldd')
        assert mapped.pop(str(lib))==lib_sha
        assert mapped==old['new_transitive_library_SHA256'] and len(mapped)==71
        binaries[arm]=dict(path=str(binary),SHA256=H(binary),bytes=binary.stat().st_size,
                           library=str(lib),library_SHA256=lib_sha,providers=len(mapped))
    receipt.update(passed=True,binaries=binaries,object_SHA256=H(obj),
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
