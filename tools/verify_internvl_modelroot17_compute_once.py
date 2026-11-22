"""Verify the mounted InternVL model on the selected Slurm training node."""

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LIB = ROOT / 'incoming/training-internvl-modelroot17-preflight-635df739726b41c29a71aceff6569076'
CURRENT = ROOT / 'results/training-internvl-modelroot17-current-mount-CPU-v1'
OUT = ROOT / 'results/training-internvl-modelroot17-compute-CPU-v1'
REQUEST = 'e90eb28e55e74ad79156039efd2f0677'
LABEL = OUT.name
PARENT, NODE = '1087656', 'a14r3n04'
PLAN_SHA = 'e2170a45e0c59112025bb2c7d8d06604f7ec7d45ec953fa2755b07d7841e1899'
LOCAL_PROOF_SHA = 'cc68e868169dd872e19cf827b2dbac5e0420293235cb55c844be4e842a71bcb1'
LIB_SHA = 'ebff2c2df8665cde980758aa0a8bfb06e5edf65cd71d1346bde2fc432315d3d6'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def load_assembler():
    path = LIB / 'assemble.py'
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != LIB_SHA:
        raise ValueError('unexpected frozen assembly verifier')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('internvl_inventory', path))
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def unescape(value):
    return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),value)


def backing_export(path, mountinfo):
    candidates = []
    path = PurePosixPath(path)
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(' - ')
        if not separator:
            continue
        fields = left.split()
        ordinary = PurePosixPath(unescape(fields[4]))
        if path.is_relative_to(ordinary):
            fs, host, *_ = right.split()
            exported = PurePosixPath(unescape(fields[3])) / path.relative_to(ordinary)
            candidates.append((len(ordinary.parts),fs,unescape(host),exported.as_posix()))
    if not candidates:
        raise ValueError('no authoritative backing mount')
    depth = max(item[0] for item in candidates)
    matches = {item[1:] for item in candidates if item[0]==depth}
    if len(matches)!=1:
        raise ValueError('ambiguous backing mount')
    fs, host, exported = matches.pop()
    if fs not in {'nfs','nfs4','lustre','beegfs','gpfs','ceph'} or not host:
        raise ValueError('original backing mount is not an identified shared filesystem')
    return sha256(json.dumps([fs,host,exported],separators=(',',':')).encode()).hexdigest()


def paths(plan):
    for name,row in sorted(plan['members'].items()):
        yield name,row,'source','stamp'
        if 'checkpoint' in row:
            yield name,row,'checkpoint','checkpoint_stamp'


def localize(a, plan, expected, mountinfo):
    observed = {}
    for name,row,path_key,stamp_key in paths(plan):
        ident = name + ':' + path_key
        a.require(expected[ident] == backing_export(row[path_key],mountinfo),'backing export differs on compute node')
        current,_ = a.fd_info(row[path_key])
        a.require(set(current)==set(row[stamp_key]) and all(current[k]==v for k,v in row[stamp_key].items() if k!='st_dev'),
                  'shared inode identity differs on compute node')
        observed[ident] = current['st_dev'] != row[stamp_key]['st_dev']
        row[stamp_key] = current
    return observed


def node(a):
    step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
    a.require(re.fullmatch(PARENT+r'\.\d+',step) and socket.gethostname().split('.')[0]==NODE
              and os.environ.get('SLURM_STEP_GPUS') in (None,'','none','N/A','(null)'),'expected the zero-GRES training node')
    plan,_ = a.document(CURRENT/'MOUNT_PLAN.json',PLAN_SHA,65536)
    anchors = json.loads((OUT/'export-pins.json').read_bytes())
    a.require(anchors['request']==REQUEST and anchors['plan_SHA256']==PLAN_SHA,'unexpected login-side export anchor')
    details = localize(a,plan,anchors['files'],Path('/proc/self/mountinfo').read_text())
    localized = OUT/'localized-plan.json'
    raw = a.serialized(plan)
    a.write_once(localized,raw)
    argv = ['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--contain','--no-home','--pwd','/']
    for bind in [*plan['binds'],f'{LIB}:/audit:ro',f'{localized}:/audit-plan.json:ro']:
        argv += ['--bind',bind]
    argv += [str(ROOT/'containers/dtk26-sandbox-r1'),'/usr/bin/python3','-I','-S','-B','/audit/assemble.py',
             'verify-mounted','--plan','/audit-plan.json','--plan-sha256',a.digest(raw)]
    child = subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=28)
    a.write_once(OUT/'container.stdout',child.stdout[-65536:])
    a.write_once(OUT/'container.stderr',child.stderr[-65536:])
    a.require(child.returncode==0,'compute-node container mount verifier failed')
    proof = json.loads(child.stdout)
    a.require(proof['state']=='VERIFIED_ALL17_READONLY_REGULAR_MOUNTS' and proof['members']==17,
              'unexpected container proof denominator')
    a.verify_originals(plan)
    result = dict(request=REQUEST,node=NODE,step=step,read_only_members=17,localized_stat_views=len(details),
                  local_device_id_differences=sum(details.values()),logical_model_bytes=proof['logical_model_bytes'],
                  localized_plan_SHA256=a.digest(raw),original_inodes_unchanged=True,
                  container_output_SHA256=a.digest(child.stdout),actual_GPU_steps=0,weight_body_bytes=0,training_READY=False)
    a.write_once(OUT/'computed-proof.json',a.serialized(result))


def controller(a):
    evidence,_ = a.document(CURRENT/'outcome.json',LOCAL_PROOF_SHA,65536)
    a.require(evidence['state']=='VERIFIED_ALL17_READONLY_ON_REHASH_HOST' and evidence['plan_SHA256']==PLAN_SHA,
              'missing exact original same-host mount proof')
    plan,_ = a.document(CURRENT/'MOUNT_PLAN.json',PLAN_SHA,65536)
    a.require(plan['verified_host_boot']==dict(host=socket.gethostname(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
              'observing host differs from hash receipt')
    a.verify_originals(plan)
    mountinfo = Path('/proc/self/mountinfo').read_text()
    files = {name+':'+path_key:backing_export(row[path_key],mountinfo) for name,row,path_key,_ in paths(plan)}
    a.require(len(files)==20,'missing model or checkpoint identity')

    def query(argv):
        return subprocess.check_output(argv,text=True,timeout=8).strip().splitlines()

    jobs = query(['scontrol','show','job',PARENT,'-o'])
    job = dict(p.split('=',1) for p in jobs[0].split() if '=' in p)
    a.require(len(jobs)==1 and job['JobState']=='RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
              and NODE in query(['scontrol','show','hostnames',job['NodeList']]),'node is not owned by the expected parent')
    for line in query(['scontrol','show','step',PARENT,'-o']):
        data = dict(p.split('=',1) for p in line.split() if '=' in p)
        if re.fullmatch(PARENT+r'\.\d+',data.get('StepId','')):
            a.require(NODE not in query(['scontrol','show','hostnames',data['NodeList']]),'existing numbered step uses the training node')

    def account():
        rows = [r.split('|') for r in query(['sacct','-nP','-j',PARENT,'-S','2026-11-19T00:00:00','-o',COLUMNS])]
        hits = [r for r in rows if len(r)==7 and r[1]==LABEL]
        a.require(len(hits)<=1,'duplicate scoped Slurm row')
        return hits[0] if hits else None

    a.require(account() is None,'original compute scope already attempted')
    OUT.mkdir(mode=0o700)
    a.write_once(OUT/'export-pins.json',a.serialized(dict(request=REQUEST,plan_SHA256=PLAN_SHA,files=files)))
    argv = ['srun','--jobid='+PARENT,'--job-name='+LABEL,'--exact','--immediate=5','-N1','-n1','--nodelist='+NODE,
            '--cpus-per-task=1','--mem=2048M','--gres=none','--time=1',sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'node']
    with (OUT/'step.log').open('xb') as f:
        child = subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=42)
    deadline = time.monotonic()+9
    row = account()
    while (not row or row[2] in {'RUNNING','PENDING','COMPLETING'}) and time.monotonic()<deadline:
        time.sleep(.5)
        row = account()
    result = dict(request=REQUEST,planned_members=17,accounting=row,accounting_columns=COLUMNS.split(','),
                  step_client_rc=child.returncode,weight_body_bytes=0,actual_GPU_steps=0,training_READY=False)
    if child.returncode==0 and row and row[2:4]==['COMPLETED','0:0'] and row[5]==NODE and 'gres/' not in row[6]:
        proof_file = OUT/'computed-proof.json'
        raw = proof_file.read_bytes()
        info = json.loads(raw)
        a.require(info['step']==row[0] and info['read_only_members']==17,'compute receipt or accounting differs')
        result.update(info,state='VERIFIED_ALL17_READONLY_ON_T03_COMPUTE_NODE',compute_proof_SHA256=a.digest(raw))
    else:
        result.update(state='COMPUTE_MOUNT_PROBE_FAILED_OR_INCOMPLETE')
    body = a.serialized(result)
    a.write_once(OUT/'outcome.json',body)
    print(json.dumps(result|{'outcome_SHA256':a.digest(body)}))
    return int(result['state']!='VERIFIED_ALL17_READONLY_ON_T03_COMPUTE_NODE')


if __name__=='__main__':
    a = load_assembler()
    a.require(os.getuid()==a.OWNER and sys.argv[1:] in ([],['node']),'fixed project-owned inspection required')
    raise SystemExit(node(a) if sys.argv[1:] else controller(a))
