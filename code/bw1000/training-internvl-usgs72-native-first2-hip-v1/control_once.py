"""Submit and account for the fixed InternVL two-update domain training once."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'training-internvl-usgs72-native-first2-hip-v1'
RUN = ROOT / 'results' / LABEL
OUT = ROOT / 'results' / (LABEL+'-controller')
PARENT, NODE, PORT = '1087656', 'a14r3n04', 29967
REQUEST = 'bc7cd25fcd6d4709a8804872fd236d8a'
COLS = ('JobIDRaw','JobName%100','State','ExitCode','ElapsedRaw','NodeList','AllocTRES')
ROLES = ('pre','gpu','post')
WHOLE, GPU_LIMIT = 1800, 1500
HEX = re.compile(r'[0-9a-f]{64}')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def put(path, doc):
    with path.open('x',encoding='utf8') as f:
        json.dump(doc,f,sort_keys=True,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def read(path, cap=65536):
    need(not path.is_symlink() and path.is_file(),'missing plain original receipt')
    with path.open('rb') as f:
        data = f.read(cap+1)
    need(0<len(data)<=cap,'receipt exceeds bounds')
    return json.loads(data)


def query(args):
    return subprocess.check_output(args,text=True,timeout=8).strip().splitlines()


def fields(row):
    return dict(item.split('=',1) for item in row.split() if '=' in item)


def placement():
    rows = query(['scontrol','show','job',PARENT,'-o'])
    need(len(rows)==1,'expected one owned parent')
    job = fields(rows[0])
    need(job['JobState']=='RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
         and 'gres/dcu=320' in job['AllocTRES'].split(',')
         and NODE in query(['scontrol','show','hostnames',job['NodeList']]),'original T03 allocation unavailable')
    for line in query(['scontrol','show','step',PARENT,'-o']):
        step = fields(line)
        if re.fullmatch(PARENT+r'\.\d+',step.get('StepId','')):
            need(NODE not in query(['scontrol','show','hostnames',step['NodeList']]),'another numbered step uses the T03 node')


def accounting(role):
    rows = [line.split('|') for line in query(['sacct','-nP','-j',PARENT,'-S','2026-11-19T00:00:00','-o',','.join(COLS)])]
    matches = [row for row in rows if len(row)==7 and row[1]==LABEL+'-'+role]
    need(len(matches)<=1,'duplicate exact numbered step')
    return matches[0] if matches else None


def resources(role, row):
    if not row or not re.fullmatch(PARENT+r'\.\d+',row[0]) or row[5]!=NODE:
        return False
    tres = dict(field.split('=',1) for field in row[6].split(','))
    return (tres.get('gres/dcu','0')==('8' if role=='gpu' else '0') and tres.get('node')=='1'
            and tres.get('cpu')==('64' if role=='gpu' else '2')
            and tres.get('mem') in (('224000M','218.75G') if role=='gpu' else ('2048M','2G')))


def cpu_inventory(role):
    import snapshot
    import owned_processes

    snapshot.NODE = NODE
    data = snapshot.snapshot(False)
    need(data['full_fd_inventory_performed'] and data['kfd_holder_count']==0 and len(data['pci'])==8,
         'expected complete clean host device inventory')
    with socket.socket() as check:
        check.bind(('127.0.0.1',PORT))
    data.update(request=REQUEST,role=role,master_port=PORT)
    if role=='post':
        final = RUN/'owned-final.json'
        exists = final.is_file()
        data['owned_registry_complete'] = exists or not (RUN/'native-intent.json').exists()
        old = read(final,2*1024**2)['ownership'] if exists else {}
        registry = {int(key):value for key,value in old.items()}
        data['owned_processes_remaining'] = len(owned_processes.refresh(registry,RUN))
    put(OUT/(role+'-inventory.json'),data)


def run_phase(role, argv, started):
    from slurm_accounting import wait_terminal

    need(accounting(role) is None,'original scope already entered this phase')
    left = (GPU_LIMIT if role=='gpu' else WHOLE) - (time.time()-started)
    seconds = min(int(left)-8,GPU_LIMIT if role=='gpu' else 60)
    minutes = seconds//60
    need(minutes>=1,'original launch epoch no longer has an admissible whole minute')
    options = ['srun','--jobid='+PARENT,'--job-name='+LABEL+'-'+role,'--exact','--immediate=6','-N1','-n1',
               '--nodelist='+NODE,'--cpu-bind='+('none' if role=='gpu' else 'threads'),
               '--cpus-per-task='+('64' if role=='gpu' else '2'),'--mem='+('224000M' if role=='gpu' else '2048M'),
               '--gres='+('dcu:8' if role=='gpu' else 'none'),'--time='+str(minutes)]
    rc = None
    with (OUT/(role+'.log')).open('xb') as log:
        try:
            rc = subprocess.run([*options,*argv],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                timeout=minutes*60+7).returncode
        except subprocess.TimeoutExpired:
            pass
    observed = wait_terminal(lambda:accounting(role),time.monotonic()+18,interval=.6,max_observations=10)
    row = observed['row']
    result = dict(observed,client_rc=rc,original_epoch=started,resource_bound=resources(role,row))
    result['passed'] = bool(rc==0 and result['resource_bound'] and row[2:4]==['COMPLETED','0:0'])
    put(OUT/(role+'-accounting.json'),result)
    return result


def run(launch):
    epoch = launch['epoch']
    self = [sys.executable,'-I','-S','-B',str(Path(__file__).resolve())]
    phases = {}
    status = dict(request=REQUEST,epoch=epoch,phases=phases,training_READY=False,formal17_qualified=False)
    try:
        phases['pre'] = run_phase('pre',[*self,'--inventory','pre'],epoch)
        need(phases['pre']['passed'],'original structured zero-GRES pre did not complete')
        before = OUT/'pre-inventory.json'
        record = read(before)
        need(record['step']==phases['pre']['row'][0] and record['node']==NODE and record['role']=='pre'
             and record['master_port']==PORT and record['request']==REQUEST,'unbound pre inventory')
        placement()
        kwargs = ['--epoch',str(epoch),'--claim-sha256',launch['claim_SHA256'],'--manifest-sha256',launch['manifest_SHA256'],
                  '--exports-sha256',launch['exports_SHA256'],'--before',str(before),'--before-sha256',sha256(before.read_bytes()).hexdigest()]
        phases['gpu'] = run_phase('gpu',[sys.executable,'-B',str(HERE/'node.py'),*kwargs],epoch)
        if phases['gpu']['terminal_visible'] and phases['gpu']['resource_bound']:
            phases['post'] = run_phase('post',[*self,'--inventory','post'],epoch)
            post_file = OUT / "post-inventory.json"
            visibility_deadline = time.monotonic() + min(10, max(0, epoch + WHOLE - time.time()))
            while phases["post"]["passed"] and not post_file.is_file() and time.monotonic() < visibility_deadline:
                time.sleep(.5)
            post = read(post_file) if phases["post"]["passed"] else {}
            status['resource_clean'] = (post.get('step')==phases['post']['row'][0] if post else False) and (
                post['boot_id']==record['boot_id'] and post['pci']==record['pci'] and post['owned_registry_complete']
                and post['owned_processes_remaining']==0 and post['master_port']==PORT)
        status['passed'] = bool(phases['gpu']['passed'] and status.get('resource_clean'))
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        status['error_type'] = type(exc).__name__
        raise
    finally:
        put(OUT/'outcome.json',status)


def submit(args):
    import native
    import node

    need(re.fullmatch(r'[0-9a-f]{40}',args.publication) and all(HEX.fullmatch(x) for x in
         (args.claim_sha256,args.manifest_sha256,args.exports_sha256)),'exact frozen source identities required')
    claim = node.bound_source(args.claim_sha256,args.manifest_sha256)
    verified = native.binding()
    need(len(verified['trainable_manifest'])==504,'original PEFT structure is unavailable before launch')
    need(args.exports_sha256 == claim["exports_SHA256"], "Exact original compute export pins required")
    placement()
    need(not RUN.exists() and not OUT.exists() and all(accounting(role) is None for role in ROLES),'original scope already exists')
    OUT.mkdir(mode=0o700)
    doc = dict(request=REQUEST,publication=args.publication,epoch=time.time(),claim_SHA256=args.claim_sha256,
               manifest_SHA256=args.manifest_sha256,exports_SHA256=args.exports_sha256)
    put(OUT/'intent.json',doc)
    with (OUT/'controller.log').open('xb') as f:
        proc = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
                                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    start = int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    output = dict(request=REQUEST,PID=proc.pid,startticks=start,state='CONTROLLER_STARTED',
                  intent_SHA256=sha256((OUT/'intent.json').read_bytes()).hexdigest())
    put(OUT/'submission.json',output)
    print(json.dumps(output))


if __name__=='__main__':
    sys.path.insert(0,str(HERE))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',choices=('pre','post'))
    p.add_argument('--run',action='store_true')
    for name in ('claim-sha256','manifest-sha256','exports-sha256','publication'):
        p.add_argument('--'+name)
    args = p.parse_args()
    if args.inventory:
        cpu_inventory(args.inventory)
    elif args.run:
        run(read(OUT/'intent.json'))
    else:
        submit(args)
