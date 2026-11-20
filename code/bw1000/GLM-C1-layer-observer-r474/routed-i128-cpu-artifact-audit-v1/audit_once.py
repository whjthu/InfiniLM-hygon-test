"""Read immutable GLM TP16 routed-I128 compiler outcomes without writing cluster state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import struct


REPO = Path(__file__).resolve().parents[4]
BRANCHES = {
    'gate': ('f13r1n09', (1, 2), 'b01e48ea24da3baf21b3231dc026cab57b2138014b6d0f795d1b0cdb3102fed2',
             'fe5a475ee9f5163f190774d89bc8b1f243170bf4cc4a8bd68f96a5753f2bcd81'),
    'down': ('f13r1n08', (2, 4), 'de6dc0b52f430d4b3eb51e0299a72a76bc90a174ff2850295bf563dfbdb77be7',
             'a8f195129cb6ae30d32c46830871d6f1c7cd801641bf5540489af300429a6495'),
}
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
EXPORTS = {'glm_r470_i128_validate', 'glm_r470_i128_gate', 'glm_r470_i128_down'}
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE',
            'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT'}
MAX_METADATA = 4 * 1024**2
MAX_OBJECT = 128 * 1024**2


def need(value, code):
    if not value:
        raise ValueError(code)


def bounded(path, maximum=MAX_METADATA):
    path = Path(path)
    with path.open('rb') as handle:
        blob = handle.read(maximum + 1)
    need(0 < len(blob) <= maximum, 'EVIDENCE_SIZE_BOUND')
    return blob


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(arm, repo=REPO):
    node, rows, manifest_sha, claim_sha = BRANCHES[arm]
    base = repo/'code/bw1000/GLM-C1-layer-observer-r474'/f'routed-i128-{arm}-rows-CPU-v1'
    task = repo/'coordination/tasks'/f'glm-i128-{arm}-rows-gfx936-cpu-v1.json'
    raw_manifest, raw_claim = bounded(base/'SOURCE_FILES.json'), bounded(task)
    need(sha(raw_manifest) == manifest_sha and sha(raw_claim) == claim_sha, 'FROZEN_BRANCH_IDENTITY')
    manifest = json.loads(raw_manifest)['files_SHA256']
    need(all(Path(name).name == name and sha(bounded(base/name)) == digest
             for name,digest in manifest.items()), 'FROZEN_SOURCE_BYTES')
    claim = json.loads(raw_claim)
    plan = json.loads(bounded(base/'PLAN.json'))
    label = f'GLM-I128-{arm}-rows-gfx936-CPU-v1'
    need((plan['label'],plan['parent'],plan['node']) == (label,'1103606',node)
         and (claim['label'],claim['parent'],claim['node']) == (label,'1103606',node)
         and (claim['CPU'],claim['memory_MiB'],claim['GPU'],claim['Slurm_seconds']) == (4,8192,0,300)
         and claim['once'] is True and claim['replay_allowed'] is False
         and claim['CPU_GO'] is True and claim['GPU_GO'] is False and claim['manifest_SHA256'] == manifest_sha,
         'FROZEN_RESOURCE_OR_SCOPE')
    need(set(plan['variants']) == {str(n) for n in rows}
         and plan['variants'] == claim['new_variants']
         and all(plan['variants'][str(n)]['source_SHA256'] == manifest[f'rows{n}.cu'] for n in rows)
         and plan['compiler_prefix'][2:] == ['-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC'],
         'FROZEN_COMPILER_OR_VARIANT')
    return plan, rows, manifest_sha, base


def inventory(repo=REPO):
    plans = [load(arm,repo) for arm in BRANCHES]
    need(len({plan['label'] for plan,_,_,_ in plans}) == len(plans)
         and len({(plan['parent'],plan['node']) for plan,_,_,_ in plans}) == len(plans), 'DUPLICATE_LAUNCH_SCOPE')
    ids = [variant['id'] for plan,_,_,_ in plans for variant in plan['variants'].values()]
    need(len(ids) == len(set(ids)) == 4, 'DUPLICATE_CANDIDATE_ID')
    return {arm:dict(label=plan['label'],node=plan['node'],parent=plan['parent'],separable_cpu_claim=True,
                     rows=list(rows),candidate_ids=[plan['variants'][str(r)]['id'] for r in rows],
                     manifest_SHA256=pin,code_path=source.relative_to(repo).as_posix(),
                     target_compiles_proven_by_inventory=0,live_admission_known=False)
            for arm,(plan,rows,pin,source) in zip(BRANCHES,plans)}


def accounting(raw, plan):
    selected = []
    for line in raw.decode('utf8').splitlines():
        if not line:
            continue
        parts = line.removesuffix('|').split('|')
        need(len(parts) == 7, 'EXACT_SLURM_COLUMNS')
        if parts[1] == plan['label']:
            selected.append(parts)
    need(len(selected) == 1, 'EXACT_SINGLE_NUMBERED_STEP')
    step,label,state,exitcode,elapsed,node,tres = selected[0]
    state_word = state.split()[0] if state.split() else ''
    alloc = dict(piece.split('=',1) for piece in tres.split(',') if '=' in piece)
    need(re.fullmatch(plan['parent']+r'\.[0-9]+',step) and state_word in TERMINAL
         and re.fullmatch(r'\d+:\d+',exitcode) and elapsed.isdecimal() and int(elapsed) <= 330
         and node == plan['node'] and alloc.get('cpu') == '4' and alloc.get('node') == '1'
         and alloc.get('mem') in {'8G','8192M','8388608K'} and not any(k.startswith('gres/') for k in alloc),
         'SLURM_TERMINAL_SCOPE')
    return selected[0]


def dependency_args(raw):
    lhs,colon,rhs = raw.decode('utf8').replace('\\\n',' ').partition(':')
    paths = set(shlex.split(rhs))
    need(bool(lhs and colon and paths) and all(path.startswith('/') for path in paths),'DEPENDENCY_EVIDENCE')
    return paths


def log(directory, command, stream):
    filename = directory/(command['label']+'.'+stream)
    with filename.open('rb') as handle:
        data=handle.read(MAX_METADATA+1)
    need(len(data) <= MAX_METADATA and sha(data) == command.get(stream+'_SHA256'), 'COMPILER_LOG_PIN')
    return data


def host_elf(data):
    # Independent structural check for the retained host ELF object. The compiler
    # argv may bind gfx936 without this proving the embedded device code object's ISA.
    need(len(data) >= 64 and data[:6] == b'\x7fELF\x02\x01' and data[6] == 1
         and struct.unpack_from('<HHI',data,16) == (1,62,1)
         and struct.unpack_from('<H',data,52)[0] == 64, 'HOST_RELOCATABLE_ELF')


def embedded_amdgpu(data):
    """Recognize bounded uncompressed AMDGPU ELF images; retain raw flags."""
    images=[]
    cursor=64
    while (offset := data.find(b'\x7fELF',cursor)) >= 0:
        cursor=offset+4
        if offset+64 > len(data) or data[offset+4:offset+7] != b'\x02\x01\x01':
            continue
        if struct.unpack_from('<H',data,offset+18)[0] != 224:
            continue
        section_table=struct.unpack_from('<Q',data,offset+40)[0]
        width,count=struct.unpack_from('<HH',data,offset+58)
        if width != 64 or not 0 < count < 4096 or section_table < 64:
            continue
        end=section_table+width*count
        if offset+end > len(data):
            continue
        for index in range(count):
            info=struct.unpack_from('<IIQQQQIIQQ',data,offset+section_table+width*index)
            if info[1] != 8:
                end=max(end,info[4]+info[5])
        if offset+end > len(data):
            continue
        images.append(dict(offset=offset,bytes=end,SHA256=sha(data[offset:offset+end]),
                           ELF_flags=struct.unpack_from('<I',data,offset+48)[0]))
        cursor=offset+end
    return images


def routed_exports(data):
    return {parts[-1] for entry in data.decode('utf8').splitlines()
            if len(parts := entry.split()) >= 3 and parts[-2] == 'T' and parts[-1].startswith('glm_r470_i128_')}


def audit(arm, launch_path, accounting_path, output_dir, publication, repo=REPO):
    plan, rows, manifest_sha, _ = load(arm,repo)
    line = accounting(bounded(accounting_path),plan)
    launch_blob = bounded(launch_path)
    launch = json.loads(launch_blob)
    epoch = launch.get('epoch')
    need(re.fullmatch('[0-9a-f]{40}',publication) and launch.get('publication') == publication
         and launch.get('manifest_SHA256') == manifest_sha
         and re.fullmatch('[0-9a-f]{32}',launch.get('request',''))
         and type(epoch) in (float,int) and math.isfinite(epoch) and epoch > 0
         and launch.get('deadline') == epoch+300 and launch.get('GPU') is False
         and str(launch.get('intent','')).endswith('/results/'+plan['label']+'-intent.json')
         and type(launch.get('pid')) is int and launch['pid'] > 0
         and launch.get('pgid') == launch['pid'] and type(launch.get('startticks')) is int
         and launch['startticks'] > 0, 'ORIGINAL_LAUNCH_BOUNDARY')
    report_path = output_dir/'CPU-result.json'
    result = dict(arm=arm,planned_variants=list(rows),accounting=line,publication=publication,
                  original_request=launch['request'],launch_SHA256=sha(launch_blob),passed=False,verified_objects=[],
                  embedded_device_ISA_independently_verified=False,device_numerical_qualified=False)
    if not report_path.is_file():
        need(line[2:4] != ['COMPLETED','0:0'], 'MISSING_SUCCESS_BUILD_REPORT')
        return dict(result,outcome='ORIGINAL_TERMINAL_WITHOUT_BUILD_REPORT')
    report_blob = bounded(report_path)
    report = json.loads(report_blob)
    need((report['original_step'],report['node'],report['epoch'],report['deadline'])
         == (line[0],plan['node'],epoch,epoch+300)
         and report['planned_new_device_TUs'] == len(rows) and report['actual_GPU_steps'] == 0
         and report['SDK_calls'] == 0 and report['weights_read'] == 0
         and report['numerical_qualified'] is False, 'CONTROLLER_REPORT_SCOPE')
    result['build_SHA256']=sha(report_blob)
    if line[2:4] != ['COMPLETED','0:0']:
        need(report['passed'] is not True,'SLURM_REPORT_TERMINAL_CONFLICT')
        return dict(result,outcome='ORIGINAL_TERMINAL_FAILED',retained_reported_rows=sorted(report['compiled']))
    need(report['completed'] is True and report['passed'] is True and report['all_fresh_M_before_compile'] is True
         and set(report['compiled']) == {str(row) for row in rows}, 'SUCCESS_REQUIRED_COMPLETE_VARIANTS')
    labels = ['original-symbols'] + [f'rows{row}-M' for row in rows]
    labels += [f'rows{row}-{phase}' for row in rows for phase in ('compile','symbols')]
    need([c['label'] for c in report['commands']] == labels and all(c.get('rc') == 0 for c in report['commands']),
         'COMPILER_COMMAND_SEQUENCE')
    commands = {cmd['label']:cmd for cmd in report['commands']}
    logs = {label:log(output_dir,cmd,'stdout') for label,cmd in commands.items()}
    for cmd in commands.values():
        log(output_dir,cmd,'stderr')
    nm = ['/usr/bin/nm','-g','--defined-only']
    need(commands['original-symbols']['argv'] == nm+[plan['original_object']]
         and routed_exports(logs['original-symbols']) == EXPORTS,'BASELINE_ABI_COMMAND')
    for row in rows:
        prefix=f'/refactor/candidates/{plan["label"]}/rows{row}/routed_i128'
        src,obj,md = (prefix+extension for extension in ('.cu','.o','.d'))
        m,cc,syms = (f'rows{row}-{phase}' for phase in ('M','compile','symbols'))
        compiler=plan['compiler_prefix']
        need(commands[m]['argv'] == compiler+['-M',src]
             and commands[cc]['argv'] == compiler+['-MD','-MF',md,'-c',src,'-o',obj]
             and commands[syms]['argv'] == nm+[obj], 'EXACT_CANDIDATE_COMMAND')
        folder=output_dir/f'rows{row}'
        source,metadata,object_bytes=(bounded(folder/'routed_i128.cu'),bounded(folder/'routed_i128.d'),
                                      bounded(folder/'routed_i128.o',MAX_OBJECT))
        pin=plan['variants'][str(row)]
        received=report['compiled'][str(row)]
        expected = dict(candidate_id=pin['id'],source_SHA256=pin['source_SHA256'],object=obj,
                        object_SHA256=sha(object_bytes),MD_SHA256=sha(metadata))
        need(all(received.get(k) == v for k,v in expected.items()) and sha(source) == expected['source_SHA256'],
             'CANDIDATE_ARTIFACT_BYTES')
        deps=dependency_args(logs[m])
        need(deps == dependency_args(metadata) and src in deps and len(deps) == received['dependency_count'],
             'FRESH_M_EQUALS_COMPILED_MD')
        need(routed_exports(logs[syms]) == set(received['host_wrapper_ABI']) == EXPORTS, 'EXACT_ROUTED_CABI')
        host_elf(object_bytes)
        result['verified_objects'].append(dict(rows=row,**expected,dependency_count=len(deps),compiled_bytes=len(object_bytes),
                                               embedded_uncompressed_AMDGPU_images=embedded_amdgpu(object_bytes)))
    return dict(result,passed=True,outcome='TWO_TARGET_COMPILER_OBJECTS_VERIFIED_WITHOUT_GPU_EXECUTION')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('inventory')
    check=sub.add_parser('audit')
    check.add_argument('arm',choices=BRANCHES)
    check.add_argument('--launch',required=True,type=Path)
    check.add_argument('--accounting',required=True,type=Path)
    check.add_argument('--output-dir',required=True,type=Path)
    check.add_argument('--publication',required=True)
    args=p.parse_args()
    if args.action=='inventory':
        print(json.dumps(inventory(),indent=2))
        return 0
    result=audit(args.arm,args.launch,args.accounting,args.output_dir,args.publication)
    print(json.dumps(result,indent=2))
    return int(not result['passed'])


if __name__=='__main__':
    raise SystemExit(main())
