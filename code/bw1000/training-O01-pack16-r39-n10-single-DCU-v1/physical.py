
from datetime import datetime,timezone
import fcntl,json,os,re,stat,subprocess,time
from pathlib import Path
from hashlib import sha256
parent,node='1103606','f12r2n10'
label='training80-o01-n10-physical-'+REQUEST[:12]
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
probe=root/'training80-admission-standby-9cfcaff1ae2342868d72861af5392acd'
ports=[43843,43844,29500]
env=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')
def q(*args):
 p=subprocess.run(args,capture_output=True,timeout=7,env=env,check=True)
 assert len(p.stdout)<1048576
 return p.stdout.decode().splitlines()
def fields(line):return dict(x.split('=',1) for x in line.split() if '=' in x)
def hosts(expr):
 return [expr] if re.fullmatch(r'f\d+r\dn\d+',expr) else q('scontrol','show','hostnames',expr)
def save(path,obj):
 with path.open('xb') as f:
    f.write((json.dumps(obj,sort_keys=True)+'\n').encode())
    f.flush()
    os.fsync(f.fileno())
state={'request':REQUEST,'node':node,'parent':parent,'GPU_steps_requested':0,'zero_GRES_invoked':False,'physical_mode':MODE,'owner_request':OWNER}
try:
    os.umask(0o077)
    assert os.getuid()==154459 and probe.resolve(strict=True)==probe
    assert all((probe/name).resolve(strict=True)==probe/name and sha256((probe/name).read_bytes()).hexdigest()==pin for name,pin in PINS.items())
    with (root/'.training80-dcu-standby.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=fields(q('scontrol','show','job','-o',parent)[0])
        assert job['JobState']=='RUNNING' and job['UserId'].endswith('(154459)') and node in hosts(job['NodeList'])
        [line]=q('squeue','-h','-w',node,'-o','%i|%T|%N')
        jid,st,places=line.split('|')
        assert (jid,st)==(parent,'RUNNING') and node in hosts(places)
        for line in q('scontrol','show','step',parent,'-o'):
            step=fields(line)
            if 'gres/dcu=' in step.get('TRES','') and step['State'] in ('RUNNING','COMPLETING','PENDING'):
                assert node not in hosts(step['NodeList'])

        cols='JobIDRaw,JobName%110,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
        old=q('sacct','-nP','-j',OLD['step'],'-o',cols)
        [account]=[line.split('|') for line in old if line.startswith(OLD['step']+'|')]
        assert len(account)==9 and account[:4]==[OLD['step'],OLD['label'],'COMPLETED','0:0']
        assert account[4].isdecimal() and int(account[4])>=OLD['seconds'] and account[5:7]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1']
        ended=datetime.strptime(account[8],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        assert OLD['expiry']<=ended+1 and ended<=time.time()
        def original(path,pin=None):
            assert path.resolve(strict=True)==path and path.stat().st_uid==os.getuid()
            with path.open('rb') as f: raw=f.read(16385)
            assert 0<len(raw)<=16384 and (pin is None or sha256(raw).hexdigest()==pin)
            return json.loads(raw),sha256(raw).hexdigest()
        worker_dir=root/('training80-dcu-standby-'+OLD['request'])
        submit,_=original(worker_dir/'submission-receipt.json',OLD['submit_SHA256'])
        expected,_=original(worker_dir/node/'expected.json',OLD['expected_SHA256'])
        worker,_=original(worker_dir/node/'worker.json',OLD['worker_SHA256'])
        cg,cg_pin=original(worker_dir/node/'worker-cgroup.json')
        end,end_pin=original(worker_dir/node/'terminal.json')
        assert submit['request']==expected['request']==OLD['request'] and submit['synthetic'] is worker['synthetic'] is True
        assert (worker['node'],worker['step'],worker['worker_pid'],worker['epoch'])==(node,OLD['step'],OLD['pid'],OLD['start'])
        assert (expected['label'],expected['seconds'],expected['parent'],expected['node'])==(OLD['label'],OLD['seconds'],parent,node)
        assert (cg['node'],cg['step'],cg['worker_pid'])==(node,OLD['step'],OLD['pid']) and cg['exact_job_and_step_cgroup'] is cg['same_UID'] is True and cg['worker_start_ticks']>0
        assert end['node']==node and end['step']==OLD['step'] and end['state']=='WORKER_RETURNED' and end['epoch']>=OLD['expiry']
        numbers=q('squeue','--steps','-h','-j',parent,'-o','%i|%j|%N')
        assert not any(re.fullmatch(re.escape(parent)+r'\.[0-9]+',x[0]) and node in hosts(x[2]) for line in numbers if len(x:=line.split('|'))==3)
        pack_label='O01-pack16-r39-n10-'+OWNER[:12]
        pack=[line.split('|') for line in q('sacct','-nP','-j',parent,'-o',cols) if len(line.split('|'))==9 and line.split('|')[1]==pack_label]
        if MODE=='pre':
            assert not pack
        else:
            [gpu]=pack
            assert gpu[5]==node and gpu[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','DEADLINE','BOOT_FAIL'}
            packed_end=datetime.strptime(gpu[8],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            assert packed_end<=time.time()
        state.update(incumbent_accounting=account,incumbent_cgroup_SHA256=cg_pin,incumbent_terminal_SHA256=end_pin,
                     old_counter_SHA256=OLD['counter_SHA256'])

        reservation=root/('training80-o01-n10-physical-'+REQUEST)
        reservation.mkdir()
        save(reservation/'intent.json',state)
        command=['srun','--jobid='+parent,'--job-name='+label,'--exact','--immediate=5',
                 '-N1','-n1','--cpus-per-task=1','--mem=256M','--gres=none','--time=00:01:00','--nodelist='+node,
                 '/usr/local/python3.12/bin/python3','-I','-S','-B','-c',PROBE,str(probe),REQUEST,parent,node,json.dumps(ports)]
        state['zero_GRES_invoked']=True
        w=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=23,env=env)
        state.update(launch_rc=w.returncode,std_err_SHA256=sha256(w.stderr).hexdigest(),stderr_bytes=len(w.stderr))
        assert w.returncode==0 and 0<len(w.stdout)<8192
        reading=json.loads(w.stdout)
        assert reading['request']==REQUEST and reading['own_probe_cgroup'] and reading['observer_uid']==154459 and reading['global_driver_KFD_queried']
        step=reading['step']
    projection=[]
    for edge in ('first','second'):
        data=reading[edge]
        assert len(set(data['PCIs']))==8 and all(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}[.][0-7]',p) for p in data['PCIs'])
        projection.append(data)
    assert all(sorted(row['PCIs'])==sorted(expected['pci']) for row in projection)
    state.update(state='ORIGINAL_ZERO_GRES_PHYSICAL_SAMPLE_CAPTURED',step=step,boot_SHA256=reading['boot_SHA256'],
                 sample=projection,other_UID_device_FDs=reading['other_UID_device_FDs'],exclusivity_lease=False)
    save(reservation/'result.json',state)
    # Read privately scored physical evidence before accounting visibility can race.
    fieldlist='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
    for delay in (0,.25,.5,1):
        time.sleep(delay)
        rows=q('sacct','-nP','-j',step,'-o',fieldlist)
        physical=[v.split('|') for v in rows if v.startswith(step+'|')]
        if len(physical)==1 and physical[0][2:4]==['COMPLETED','0:0']:
            break
    state['Slurm_accounting_observed']=physical
    state['raw_child_stdout_SHA256']=sha256(w.stdout).hexdigest()
    save(reservation/'accounting-read.json',state)
except (OSError,ValueError,KeyError,AssertionError,subprocess.SubprocessError) as exc:
    state.update(state='PHYSICAL_READ_REQUIRES_RECONCILIATION',error_type=type(exc).__name__)
print(json.dumps(state,sort_keys=True),flush=True)
