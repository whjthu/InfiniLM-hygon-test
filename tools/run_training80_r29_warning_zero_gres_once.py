"""Diagnose one repeated R29 stderr fingerprint using exactly one new zero-GRES step."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r29-warning-zero-gres-v1'
WRAPPER = ROOT / 'tools/run_training80_r29_warning_zero_gres_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
LAST40 = ROOT / 'tools/run_training80_r25_last40_ordinary_guarded_refill_once.py'
LAST40_SHA = '575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e'
CLASSIFIER = ROOT / 'tools/run_training80_prebridge_two_r2_guarded_refill_once.py'
CLASSIFIER_SHA = 'ae167c7c98db4ef598beec527e7797c7f781fcec6f52ca11c96d4ab5b39b257f'
FOOTER = LOCAL / 'training80-r29-last40-exact-first-controller-reconstructed-footer.json'
FOOTER_SHA = '2c3db4946e667edb1c7551f4e2a0a812db4e45fc5ba3d8fefa96cf048ff446a9'
NODE, PARENT, PREVIOUS = 'f12r1n16', '1194900', '1194900.1317'
WITNESSES = (NODE, 'f12r3n06')
REQUEST = '1a836ae5987c47389862adf0b3703417'
OLD_STDERR = '819487f21be5e7eeffc86fdd45d5d28f64adbecc596ef82ba176c8a5bc4b6799'
OLD_DETAIL = '65f3a77095fdcab49b34702824d24e5e44299f56107c36f5985aec601edb2c5b'
GPU_TRES = 'cpu=8,gres/dcu=8,mem=16G,node=1'


HOST_DIAG = r'''
def file_json(path):
    require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(), 'original receipt owner or path differs')
    with path.open('rb') as handle:
        raw=handle.read(8193)
    require(0<len(raw)<=8192,'original receipt exceeds the bound')
    return json.loads(raw)


def write_raw(path, raw):
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def labels(names):
    all_rows=query('sacct','-nP','-j',PARENT,'-S','2026-11-25T06:00:00','-o','JobIDRaw,JobName%110')
    found={name:[] for name in names}
    for row in all_rows:
        step,sep,label=row.partition('|')
        if sep and label in found:
            require(STEP.fullmatch(step), 'unbound prior or diagnostic Slurm step')
            found[label].append(step)
    return found


def terminal_cpu(number,label):
    row=account(number)
    require(row and row[:6]==[number,label,'COMPLETED','0:0',NODE,ZERO]
            and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',row[6]),
            'zero-GRES original accounting lacks the exact C0/End')
    return row


def project(sample):
    require(sample['own_probe_cgroup'] is True and sample['global_driver_KFD_queried'] is True
            and sample['other_UID_device_FDs']=='UNOBSERVED','zero-GRES UID or global KFD evidence unavailable')
    points=[]
    for label in ('first','second'):
        point=sample[label]
        require(sorted(point['PCIs'])==PCI and len(set(point['PCIs']))==8
                and all(type(point[name]) is int and point[name]>=0 for name in ('kfd_process_count','own_device_fd_process_count'))
                and type(point['occupied_requested_ports']) is list and set(point['occupied_requested_ports']) <= set(PORTS),
                'physical identity or counts differ')
        points.append(dict(PCIs=sorted(point['PCIs']),KFD_holders=point['kfd_process_count'],
                           project_UID_device_FD_holders=point['own_device_fd_process_count'],occupied_ports=point['occupied_requested_ports']))
    return dict(samples=points,own_probe_cgroup=True,global_KFD_observed=True,other_UID_device_FDs='UNOBSERVED',
                project_UID=sample['observer_uid'],boot_SHA256=sample['boot_SHA256'])


def classify_stderr(raw):
    data=diagnose_stderr(raw)
    lines=[r for r in data['redacted_lines'] if r['source']!='unrecognized' or r['severity']!='unrecognized' or r['terms']]
    if not raw:
        kind='EMPTY'
    elif lines and not data['classification_truncated'] and len(lines)==len(data['redacted_lines']) and all(
            r['source'] in ('srun','slurmstepd') and r['severity']=='warning' for r in lines):
        kind='SLURM_WARNING'
    elif any(r['source'] in ('srun','slurmstepd') and r['severity'] in ('error','fatal') for r in lines):
        kind='SLURM_ERROR_OR_FATAL'
    else:
        kind='UNKNOWN_NONEMPTY_STDERR'
    data.update(standard_type=kind, matches_prior_rc0_stderr=data['exact_SHA256']==OLD_STDERR,
                allows_any_GPU_or_standby_admission=False)
    return data


def main():
    os.umask(0o077)
    result=dict(request=REQUEST,node=NODE,parent=PARENT,state='PRECHECK',reserved_GPU=0,
                GPU_srun_attempted=False,zero_GRES_call_attempted=False,replay_allowed=False,GPU_admission_eligible=False)
    root=RESULTS/SCOPE
    root.mkdir(mode=0o700)
    try:
        require(os.getuid()==154459,'login UID differs')
        require(all((PROBE/name).resolve(strict=True)==PROBE/name and sha256((PROBE/name).read_bytes()).hexdigest()==bound
                    for name,bound in PROBE_PINS.items()),'frozen deployed zero-GRES source differs')
        global dcu_count
        dcu_count=runpy.run_path(str(PROBE/'read_training_slurm_membership.py'))['allocated_dcu']
        with (RESULTS/'.training80-dcu-standby.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            durable=file_json(RESULTS/OLD_SCOPE/'outcome.json')
            require(durable==OLD_BODY,'durable original authenticated GPU-not-sent receipt differs')
            require(not os.path.lexists(RESULTS/('training80-dcu-standby-'+OLD_REQUEST)), 'original GPU directory exists')
            keys=labels((OLD_LABEL,OLD_GPU_LABEL,LABEL))
            require(len(keys[OLD_LABEL])==1 and not keys[OLD_GPU_LABEL] and not keys[LABEL], 'old probe accounting missing, duplicate or another original is in place')
            result['original_zero_GRES_accounting']=terminal_cpu(keys[OLD_LABEL][0],OLD_LABEL)
            predecessor=account(PREVIOUS)
            require(predecessor and predecessor[:6]==[PREVIOUS,PREVIOUS_LABEL,'COMPLETED','0:0',NODE,'cpu=8,gres/dcu=8,mem=16G,node=1']
                    and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',predecessor[6]),
                    'original ordinary predecessor terminal changed')
            scope()
            no_foreign_allocation()
            result['zero_GRES_call_attempted']=True
            save(root/'zero-gres-intent.json',result)
            command=['srun','--jobid='+PARENT,'--job-name='+LABEL,'--exact','--overlap','--immediate=5',
                     '--nodes=1','--ntasks=1','--ntasks-per-node=1','--cpus-per-task=1','--mem=256M','--gres=none',
                     '--time=00:01:00','--kill-on-bad-exit=0','--nodelist='+NODE,
                     '/usr/local/python3.12/bin/python3','-I','-S','-B','-c',NODE_PROBE,str(PROBE),REQUEST,PARENT,NODE,json.dumps(PORTS)]
            probe=run(command,18)
            write_raw(root/'stderr.private.raw',probe.stderr)
            write_raw(root/'stdout.private.raw',probe.stdout)
            result.update(probe_rc=probe.returncode,stderr_projection=classify_stderr(probe.stderr),
                          stdout_bytes=len(probe.stdout),stdout_SHA256=sha256(probe.stdout).hexdigest())
            require(0<len(probe.stdout)<=8192,'new diagnostic stdout has no bounded physical receipt')
            sample=json.loads(probe.stdout)
            step=bind_probe(sample)
            result['new_zero_GRES_step']=step
            result['physical_evidence']=project(sample)
            for index in range(4):
                row=account(step)
                if row and row[2]=='COMPLETED':
                    break
                require(time.monotonic()+.25<DEADLINE,'new diagnostic accounting timed out')
                time.sleep(.25)
            result['new_zero_GRES_accounting']=terminal_cpu(step,LABEL)
            scope(own=step)
            no_foreign_allocation()
            require(probe.returncode==0,'new zero-GRES subprocess nonzero')
            result['state']='DIAGNOSTIC_C0_AND_PHYSICAL_OBSERVED'
    except (OSError, ValueError, TypeError, KeyError, AssertionError, subprocess.SubprocessError) as error:
        result.update(state='DIAGNOSTIC_UNQUALIFIED_RECONCILE',detail_class=type(error).__name__,
                      detail_SHA256=sha256(str(error).encode()).hexdigest())
    save(root/'outcome.json',result)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)

main()
'''


def digest(raw):
    return sha256(raw).hexdigest()


def check(value, why):
    if not value:
        raise ValueError(why)


def pin(path, sha=None):
    check(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'exact source or receipt path differs')
    raw=path.read_bytes()
    check(0<len(raw)<=262144 and (sha is None or digest(raw)==sha),'exact source or receipt bytes differ')
    return raw


def classify_originals():
    footer=json.loads(pin(FOOTER,FOOTER_SHA))
    grouped={}
    for node,rec in footer['receipt_by_node'].items():
        if rec['state']!='GPU_NOT_SENT':
            continue
        request=rec['request']
        path=LOCAL/f'training80-r25-last40-ordinary-guarded-bf16-v1-{node}-{request}-reply.json'
        reply=json.loads(pin(path,rec['SHA256']))
        transport=reply['transport']
        body=transport['response']
        check(transport['remote_outcome']=='RESPONSE_OBSERVED' and transport['authority_verified'] is transport['remote_received'] is True
              and transport['rc']==0 and reply['request']==transport['request']==body['request']==request
              and body['node']==node and body['state']=='GPU_NOT_SENT' and body['GPU_srun_attempted'] is False
              and body['zero_GRES_call_attempted'] is True and body['probe_rc']==0 and body['replay_allowed'] is False
              and body['probe_stderr_SHA256']==OLD_STDERR and body['detail_SHA256']==OLD_DETAIL
              and 'probe_step' not in body,'one of the twelve old outcomes differs')
        grouped[node]=(rec['SHA256'],body)
    check(len(grouped)==12 and all(n in grouped for n in WITNESSES),'bounded original fingerprint scope differs')
    return grouped


def scope(draft=False):
    original=classify_originals()
    path=DRAFT if draft and not CLAIM.exists() else CLAIM
    raw=pin(path)
    claim=json.loads(raw)
    check(claim==dict(scope=STEM,state='PREPARED_ONLY_ONE_ZERO_GRES_DIAGNOSTIC',request=REQUEST,node=NODE,parent=PARENT,
                      predecessor=PREVIOUS,CPU=1,memory_MiB=256,Slurm_step_seconds=60,GPU=0,max_invocations=1,replay_allowed=False,
                      original_footer_SHA256=FOOTER_SHA,original_stderr_SHA256=OLD_STDERR,original_detail_SHA256=OLD_DETAIL,
                      witness_reply_SHA256={n:original[n][0] for n in WITNESSES},original_manifest_matched_nodes=12,
                      launcher=WRAPPER.relative_to(ROOT).as_posix(),raw_stdio_remote_only=True,GPU_admission_eligible=False),
          'new diagnostic claim differs')
    pin(LAST40,LAST40_SHA)
    tool=runpy.run_path(str(LAST40));api=tool['bound_api']();ctx=api['load']()
    row=ctx[-2][NODE]
    check(row['parent']==PARENT and row['previous']==PREVIOUS and row['new_request']==original[NODE][1]['request'],
          'original ordinary guard identity differs')
    return original,api,ctx,row,digest(raw)


def render(material,revision):
    original,api,ctx,row,_=material
    previous_label=f"training80-dcu-standby-{row['request'][:12]}-{NODE}"
    placeholder=[PREVIOUS,previous_label,'COMPLETED','0:0','0',NODE,GPU_TRES,'','2026-11-25T00:00:00']
    audit_source=api['render'](ctx,row,placeholder,'0'*64,revision)['launch']
    tree=ast.parse(audit_source)
    constants={'PARENT','NODE','PREVIOUS','PREVIOUS_LABEL','PORTS','PCI','OBSERVER_STEP','OBSERVER_LABEL','ALLOW_OBSERVER_OVERLAP',
               'PROBE_PINS','NODE_PROBE','RESULTS','PROBE','TERMINAL','STEP','HOSTNAME','ZERO','FMT','DEADLINE','CLEAR_STEP'}
    functions={'require','run','query','fields','account','scope','save','no_foreign_allocation','bind_probe'}
    selected=[obj for obj in tree.body if (isinstance(obj,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in obj.targets))
              or (isinstance(obj,ast.FunctionDef) and obj.name in functions)]
    check(len(selected)==len(constants)+len(functions),'deployed zero-GRES observer extraction differs')
    pin(CLASSIFIER,CLASSIFIER_SHA)
    source=runpy.run_path(str(CLASSIFIER))['HOST_ADAPTER']
    [classifier]=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='diagnose_stderr']
    old=original[NODE][1]['request']
    values=dict(REQUEST=REQUEST,SCOPE=STEM+'-'+REQUEST,LABEL='training80-r29-warning-'+REQUEST[:12],
                OLD_SCOPE=f'training80-r25-last40-ordinary-guarded-bf16-v1-{NODE}-{old}',
                OLD_LABEL='training80-r25-fresh-'+old[:12],OLD_GPU_LABEL=f'training80-dcu-standby-{old[:12]}-{NODE}',
                OLD_REQUEST=old,OLD_BODY=original[NODE][1],OLD_STDERR=OLD_STDERR)
    header=api['UTC']+'import fcntl,json,pathlib,re,runpy,subprocess,time\nfrom pathlib import Path\nfrom hashlib import sha256\n'
    header+='\n'.join(f'{name}={value!r}' for name,value in values.items())+'\n'
    host=header+ast.unparse(ast.Module(body=[*selected,classifier],type_ignores=[]))+'\n'+HOST_DIAG
    tree=ast.parse(host)
    check(not re.search(r'controller\s*\[|subprocess\.Popen|--gres=dcu:',host) and host.count("'--gres=none'")==1
          and not any(getattr(n,'name','')=='run_guarded' for n in tree.body), 'diagnostic unexpectedly contains a GPU launcher')
    compile(host,'<one repeating-warning zero-GRES diagnostic>','exec')
    return host


def fixture(host):
    names={'classify_stderr','diagnose_stderr'}
    funcs=[n for n in ast.parse(host).body if isinstance(n,ast.FunctionDef) and n.name in names]
    sandbox=dict(re=re,sha256=sha256,OLD_STDERR=OLD_STDERR)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<actual warning projection>','exec'),sandbox)
    classify=sandbox['classify_stderr']
    for raw,kind in [(b'','EMPTY'),(b'srun: warning: CPU binding /sensitive/path NAME\n','SLURM_WARNING'),
                     (b'slurmstepd: error: step cgroup permission denied\n','SLURM_ERROR_OR_FATAL'),
                     (b'opaque /sensitive/path NAME\n','UNKNOWN_NONEMPTY_STDERR')]:
        d=classify(raw)
        check(d['standard_type']==kind and d['exact_SHA256']==digest(raw) and d['allows_any_GPU_or_standby_admission'] is False
              and '/sensitive/path' not in json.dumps(d) and 'NAME' not in json.dumps(d),'actual bounded classifier fixture differs')


def run_once(material,revision,host):
    original,api,ctx,row,claim_sha=material
    check(Path(__file__).resolve()==WRAPPER and re.fullmatch(r'[a-f0-9]{40}',revision),'exact diagnosed scope must be published')
    api['publication'](ctx,revision)
    _,publisher=ctx[3][-2]['modules']()
    for path in (WRAPPER,CLAIM,CLASSIFIER):
        check(publisher['git']('cat-file','blob',revision+':'+path.relative_to(ROOT).as_posix())==pin(path),
              'published diagnostic input differs')
    api['exact_no_other_claims'](ctx[2],ctx[3],row)
    check(NODE not in ctx[0]['local_holds'](),'configured business work retains priority')
    intent,token,reply=(LOCAL/f'{STEM}-{REQUEST}-{suffix}.json' for suffix in ('intent','send','reply'))
    check(not any(path.exists() for path in (intent,token,reply)),'original diagnostic already reserved; reconcile')
    helper=runpy.run_path(str(ctx[1][0]['HELPER']))
    sender=runpy.run_path(str(ctx[1][0]['TRANSPORT']))
    argv=helper['ssh'](ROOT.parent)
    record=dict(request=REQUEST,publication=revision,claim_SHA256=claim_sha,program_SHA256=digest(host.encode()),GPU=0,replay_allowed=False)
    ctx[1][1]['save'](intent,record)
    observed=sender['send'](argv,request=REQUEST,program=host,data=b'',send_token=token,authority_seconds=18,outcome_seconds=75)
    ctx[1][1]['save'](reply,dict(record,transport=observed))
    body=observed.get('response') or {}
    return dict(receipt=reply.name,SHA256=digest(pin(reply)),state=body.get('state'),step=body.get('new_zero_GRES_step'),
                GPU_admission_eligible=False,replay_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--publication')
    args=parser.parse_args()
    material=scope(draft=args.check)
    program=render(material,args.publication)
    fixture(program)
    if args.check:
        print(json.dumps(dict(Git=0,SCNet=0,Slurm=0,GPU=0,matched_original_guards=12,claim_SHA256=material[-1],
                             exact_host_SHA256=digest(program.encode()),classification_fixtures=4,request=REQUEST),sort_keys=True))
    else:
        print(json.dumps(run_once(material,args.publication,program),sort_keys=True))


if __name__=='__main__':
    main()
