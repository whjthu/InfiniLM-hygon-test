"""Independent original169 receipt/MD/passive-log audit; no build or library load."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W = R / 'results/GLM474-attention-detail-CPU-v1'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def main():
    evidence = {}
    started = time.monotonic()

    def original(path):
        assert time.monotonic() - started < 85
        assert path.is_file() and not path.is_symlink()
        b = path.read_bytes()
        evidence[str(path.relative_to(W))] = dict(bytes=len(b), SHA256=hashlib.sha256(b).hexdigest())
        return b

    def read(name):
        return json.loads(original(W / name))

    def md(name):
        target, separator, body = original(W / name).decode().replace('\\\n', ' ').partition(':')
        assert separator and target
        return set(shlex.split(body))

    def host(path):
        assert path.startswith('/refactor/')
        return R / path[len('/refactor/'):]

    intent, outcome = read('intent.json'), read('outcome.json')
    assert outcome['passed'] and outcome['rc'] == 0 and not outcome['client_timeout']
    assert intent['publication'] == 'd7fe8c2672a53ad56288cc6f0aa38dd90ad4fa4b'
    assert intent['deadline'] == intent['epoch'] + 600
    assert intent['work_deadline'] == intent['epoch'] + 430
    assert intent['inside_deadline'] == intent['epoch'] + 450
    columns = subprocess.check_output(['sacct', '-n', '-P', '-j', '1103606.169', '-o', COLUMNS],
                                      timeout=10, text=True)
    rows = [line.split('|') for line in columns.splitlines() if line.split('|')[0] == '1103606.169']
    assert len(rows) == 1
    row = rows[0]
    assert row == outcome['accounting']['row'] and row[1:4] == ['GLM474-attention-detail-CPU-v1', 'COMPLETED', '0:0']
    assert row[5] == 'f13r1n05' and set(row[6].split(',')) == {'cpu=4', 'mem=16G', 'node=1'}
    assert hashlib.sha256(original(W/'src/SOURCE_FILES.json')).hexdigest() == '0f013fec2e99639ab855e1d15d0c4b75fee4f27a4132a688522f71670ba7fa4a'
    manifest = read('src/SOURCE_FILES.json')
    for name, pin in manifest['files'].items():
        raw = original(W/'src'/name)
        assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256']
    build = read('build-result.json')
    assert evidence['build-result.json']['SHA256'] == outcome['build_SHA256'] == 'dda690c7ae07180f3b58ae4039c73b9f53d13c6726d96f53c7fab07ef6f64cec'
    assert build['passed'] and build['step'] == row[0] and build['new_HOST_TUs'] == 3
    assert build['new_GPU_TUs'] == 0 and build['reused_objects'] == 29 and build['linked_objects'] == 32
    plan = read('src/BUILD_PLAN.json')
    commands = build['commands']
    names = [c['name'] for c in commands]
    assert names[:3] == ['nov-M', 'prepare-M', 'output-M']
    for command in commands:
        assert command['rc'] == 0
        for stream in ('stdout', 'stderr'):
            raw = original(W/(command['name']+'.'+stream))
            assert hashlib.sha256(raw).hexdigest() == command[stream+'_SHA256']
    dependencies = {}
    for unit in plan['units']:
        name = unit['name']
        paths = md(name+'.before.d')
        assert paths == md(name+'.actual.d') == set(build['fresh_M'][name]['dependencies_SHA256'])
        allowed = dict(unit['optional_pinned_system_dependencies'], **unit['required_dependencies'])
        assert set(unit['required_dependencies']) <= paths <= set(allowed)
        assert all(build['fresh_M'][name]['dependencies_SHA256'][p] == allowed[p] for p in paths)
        assert build['MD'][name]['equal_fresh_M'] and build['MD'][name]['count'] == len(paths)
        assert names.index(name+'-compile') >= 3
        compile_command = next(c for c in commands if c['name'] == name+'-compile')['argv']
        assert compile_command == unit['compiler_prefix'] + ['-MD','-MF',str(Path('/refactor/results/GLM474-attention-detail-CPU-v1')/(name+'.actual.d')),'-MT',unit['object'],'-c',unit['source'],'-o',unit['object']]
        raw = original(host(unit['object']))
        assert hashlib.sha256(raw).hexdigest() == build['object_SHA256'][name]
        dependencies[name] = len(paths)
    link = next(c for c in commands if c['name'] == 'link')['argv']
    assert link == plan['link_argv'] and len([x for x in link if x.endswith('.o')]) == 32
    library = host(link[link.index('-o')+1])
    raw = original(library)
    assert hashlib.sha256(raw).hexdigest() == build['library_SHA256'] and len(raw) == build['library_bytes']
    def strong(label):
        return {x[-1]:x[-2] for line in (W/(label+'.stdout')).read_text().splitlines()
                if len(x:=line.split()) >= 3 and x[-2] in 'TDBR'}
    assert strong('candidate-nm') == strong('parent-nm') and len(strong('candidate-nm')) == 280
    for label in ('parent-ldd', 'candidate-ldd'):
        log = (W/(label+'.stdout')).read_text()+'\n'+(W/(label+'.stderr')).read_text()
        assert 'not found' not in log and 'undefined symbol' not in log
        providers = set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)', log, re.M))
        assert providers == set(plan['providers']) and len(providers) == 71
    owner, observer = 'glm_selected_value_layer0::detail::decode_owner', 'glm_sparse_r131::current'
    for label, symbols in [('nov', (owner, observer)), ('prepare', (observer,)),
                           ('output', (observer,)), ('bridge', (owner,)), ('linked', (owner, observer))]:
        table = (W/(label+'-tls.stdout')).read_text()
        for name in symbols:
            rows = [line.split() for line in table.splitlines() if line.rstrip().endswith(' '+name)]
            assert len({(x[1],x[6]) for x in rows if x[3]=='TLS' and x[6]!='UND'}) == 1
    for label in ('parent-elf', 'candidate-elf'):
        table = (W/(label+'.stdout')).read_text()
        assert 'SYMBOLIC' in table
        assert re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]', table) == plan['NEEDED']
    assert build['tls_storage_verified'] and build['Native_dlopen'] is False and build['GPU_GO'] is False
    result = dict(state='ORIGINAL169_INDEPENDENT_CPU_BUILD_AUDIT_PASS', accounting_columns=COLUMNS,
                  accounting=row, fresh_M_MD_counts=dependencies, new_HOST_TUs=3, new_GPU_TUs=0,
                  reused_objects=29, parent_qualification_reused='1103606.143', library_SHA256=build['library_SHA256'],
                  library_bytes=build['library_bytes'], strong_exports=280, providers=71,
                  original_passive_logs_verified=True, provider_file_hashes_recomputed=False,
                  unchanged_parent_objects_rehashed=False, originals=evidence, Native_dlopen=False,
                  GPU_GO=False, Model_GO=False, elapsed_seconds=time.monotonic()-started)
    output = R/'results/GLM474-attention-detail-CPU169-root-reader-v1.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
