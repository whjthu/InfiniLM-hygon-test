"""Four explicitly assigned single-stage srun processes across two parents."""
import json,os,re,signal,subprocess,time,math
from pathlib import Path
from contract import STAGES,TERMINAL,GPU_SECONDS


def clean_env():
    return {k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}


class Backend:
    def __init__(self,root,directory,run):
        self.root,self.directory,self.run=Path(root),Path(directory),run
        self.env=clean_env();self.processes={};self.actual={};self.ignored=set()
    def label(self,stage):return 'GLM-r361-'+self.run+'-s%d'%stage
    def command(self,args):return subprocess.check_output(args,env=self.env,text=True,timeout=25)
    def nodes(self,s):return self.command(['scontrol','show','hostnames',s]).splitlines()
    def parents(self):
        for p in {p for p,_ in STAGES}|{910407}:
            raw=self.command(['scontrol','show','job',str(p),'-o'])
            assert 'UserId=aclsylqidf('in raw and'JobState=RUNNING'in raw
            m=re.search(r'(?:^| )NodeList=([^ ]+)',raw);assert m
            required={n for parent,n in STAGES if parent==p}if p!=910407 else {'a14r2n20'}
            assert required<=set(self.nodes(m.group(1)))
    def rows(self,parent):
        raw=self.command(['sacct','-S','2026-11-01','-j',str(parent),'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%120'])
        return [x.split('|')for x in raw.splitlines()if x.strip()]
    def account(self,stage):
        parent,node=STAGES[stage];rows=[r for r in self.rows(parent)if r[1]==self.label(stage)]
        assert len(rows)<=1,'Duplicate exact stage label is never retry authority'
        if not rows:return None
        row=rows[0]
        assert re.fullmatch(str(parent)+r'\.\d+',row[0])and self.nodes(row[4])==[node],'Stage identity mismatch'
        assert self.actual.get(stage,row[0])==row[0],'Stage handle changed'
        self.actual[stage]=row[0];return row
    def no_foreign(self):
        for p in {x[0]for x in STAGES}:
            own_nodes={n for pp,n in STAGES if pp==p}
            for row in self.rows(p):
                if not re.fullmatch(str(p)+r'\.\d+',row[0])or row[0]in self.ignored or row[0]in self.actual.values():continue
                if row[2].split()[0].rstrip('+')not in TERMINAL:
                    assert not set(self.nodes(row[4]))&own_nodes,'Active foreign step on r355 nodes'
    def probe(self,suffix):
        reports=[]
        for stage,(parent,node)in enumerate(STAGES):
            args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+self.label(stage)+'-'+suffix,'python3','-B',str(self.directory/'resource_probe.py')]
            p=subprocess.run(args,env=self.env,text=True,capture_output=True,timeout=150)
            rows=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')]
            assert p.returncode==0 and len(rows)==1,'Assigned fresh/post resource probe failed: '+p.stderr[-1000:]
            q=rows[0];assert q['node']==node and q['CPU_only']and not q['GPU_initialized']and q['KFD_empty']and q['port43843_bindable']
            assert re.fullmatch(str(parent)+r'\.\d+',q['actual_step']);self.ignored.add(q['actual_step']);reports.append(q)
        return reports
    def start(self,stage,epoch,owner_sha):
        assert stage not in self.processes,'No retry or second residency'
        parent,node=STAGES[stage];left=epoch+GPU_SECONDS-time.time();assert left>180
        wall='%02d:%02d:%02d'%(int(left)//3600,int(left)%3600//60,int(left)%60)
        env=dict(self.env,WAVE_INSIDE_R2323='GLM-prefill-window-interleave-r361/runtime-r1/owner/GPU_inside_r290.sh',
            GLM_STAGE_R5040=str(stage),GLM_R290_GPUSTART_UNIX=str(epoch),GLM_SOURCE_MANIFEST_SHA=owner_sha,
            GLM_RUN_R5040=self.run,GLM_R361_CANCEL_FILE='/refactor/control/GLM-r361-cancel-'+self.run+'.json',
            GLM_SHARED_PREFIX_VALIDATE_R5814='0',GLM_PREFLIGHT_ONLY_R5194='0',GLM_DECODE_ROW_DEDUP_R5386='0',
            GLM_BOUNDED_ROPE_R4279='1',GLM_SHARED_CONTROL_R4279='1',GLM_PP_SIZE_R3603='4',GLM_CACHE_ROWS_R3627='2',GLM_CACHE_CAP_R3627='32768',
            GLM_NETWORK_TRIAL='shca',GLM_PROJECTED_HOST_R4504='0',GLM_PIPELINE_TAIL_NODE='a02r4n05',
            GLM_EXACT_BANKS_R4188='1',GLM_CHANNELS_R4363='0',GLM_VARIANT_R4363='layer77-5783',
            INFINILM_OWNER_CPU_WIDTH_R2392='4',USE_WEIGHT_CACHE_R3604='0',GLM_SERVICE_HOST_R143='127.0.0.1',GLM_SERVICE_PORT_R143='43843',
            GLM_REGULAR_FIXTURE_RECEIPT_R180='/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json',
            GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180='ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf')
        args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c64','-w',node,'--gres=dcu:8','--mem=128000M','-t',wall,'--job-name='+self.label(stage),'--output='+str(self.root/'logs'/(self.label(stage)+'-%J.log')),'--export=ALL','bash',str(self.directory/'resource_wave_host_alias_outer.sh'),'probe']
        with(self.root/'logs'/(self.label(stage)+'-client.log')).open('x')as log:
            p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.processes[stage]=p
        return dict(stage=stage,parent=parent,node=node,label=self.label(stage),PID=p.pid)
    def observe(self,stage):return dict(row=self.account(stage),srun_exit=self.processes[stage].poll())
    def cancel(self):
        records=[]
        for stage in self.processes:
            try:
                row=self.account(stage)
                if row and row[2].split()[0].rstrip('+')not in TERMINAL:
                    subprocess.run(['scancel','--signal=TERM',row[0]],env=self.env,check=True,timeout=25);records.append(dict(stage=stage,step=row[0],signal='TERM'))
            except BaseException as error:records.append(dict(stage=stage,error=repr(error)))
        for stage,p in self.processes.items():
            for sig in [signal.SIGTERM,signal.SIGKILL]:
                try:os.killpg(p.pid,sig)
                except ProcessLookupError:break
                records.append(dict(stage=stage,owned_PGID=p.pid,signal=int(sig)))
                try:p.wait(timeout=20 if sig==signal.SIGTERM else 10)
                except subprocess.TimeoutExpired:continue
                try:os.killpg(p.pid,0)
                except ProcessLookupError:break
        return records
