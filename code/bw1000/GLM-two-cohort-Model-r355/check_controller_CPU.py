"""Actual4-stage backend argv/identity and actual controller loop; no Slurm/GPU."""
import sys,json,tempfile,time,os,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D/'controller'))
import controller,backend,contract


def rejects(fn):
    try:fn()
    except(AssertionError,ValueError,RuntimeError,TimeoutError,InterruptedError):return
    raise AssertionError('Expected rejection')


def argv_and_identity():
    with tempfile.TemporaryDirectory()as tmp:
        root=Path(tmp);(root/'logs').mkdir();b=backend.Backend(root,D/'controller','cpu-r355-check');calls=[]
        def popen(args,**kw):
            calls.append((args,kw));return SimpleNamespace(pid=9000+len(calls),poll=lambda:None,wait=lambda timeout:0)
        with patch.object(backend.subprocess,'Popen',side_effect=popen),patch.object(backend.time,'time',return_value=1000):
            for stage in range(4):b.start(stage,1000,'a'*64)
            rejects(lambda:b.start(0,1000,'a'*64))
        assert len(calls)==4
        for stage,(args,kw)in enumerate(calls):
            parent,node=contract.STAGES[stage]
            assert '--jobid='+str(parent)in args and args[args.index('-w')+1]==node
            assert '-N1'in args and '-n1'in args and '--gres=dcu:8'in args
            assert kw['env']['GLM_STAGE_R5040']==str(stage) and kw['env']['GLM_CACHE_ROWS_R3627']=='2'
            assert kw['env']['GLM_PIPELINE_TAIL_NODE']=='a02r4n05' and kw['start_new_session']
            assert args[-1]=='probe'and args[args.index('-t')+1]=='02:00:00'
        for stage,(parent,node)in enumerate(contract.STAGES):
            good=[str(parent)+'.100',b.label(stage),'RUNNING','0:0',node]
            bb=backend.Backend(root,D/'controller',b.run);bb.rows=lambda p,r=good:[r];bb.nodes=lambda s:[s]
            assert bb.account(stage)==good
            wrong=good.copy();wrong[4]='a02r4n01';bb.rows=lambda p,r=wrong:[r]
            rejects(lambda:bb.account(stage))
    return dict(stages=4,parents=[910371,910374],explicit_stage_mapping=True,wrong_node_rejected=True)


class Clock:
    def __init__(self):self.t=1000
    def now(self):return self.t
    def sleep(self,n):self.t+=n


class Recording:
    def __init__(self,c,scenario):self.clock=c;self.scenario=scenario;self.processes={};self.actual={};self.started=[];self.probes=[];self.canceled=False
    def parents(self):pass
    def account(self,s):return None
    def no_foreign(self):pass
    def probe(self,suffix):self.probes.append(suffix);return [dict(stage=s,KFD_empty=True)for s in range(4)]
    def start(self,s,epoch,sha):
        assert s not in self.processes;self.processes[s]=True;self.started.append(s)
        if self.scenario=='spawn_failure'and s==2:raise OSError('recording stage2 spawn failed')
        return dict(stage=s,parent=contract.STAGES[s][0],node=contract.STAGES[s][1])
    def observe(self,s):
        self.actual[s]=str(contract.STAGES[s][0])+'.'+str(100+s)
        row=[self.actual[s],'stage'+str(s),'RUNNING','0:0',contract.STAGES[s][1]];code=None
        if self.canceled:row[2:4]=['CANCELLED','0:15'];code=15
        elif self.scenario=='stage_failure'and s==1:row[2:4]=['FAILED','1:0'];code=1
        elif self.scenario=='account_lag'and self.clock.t<1006:row=None
        elif self.scenario=='exit_unknown':row=None;code=0
        elif self.scenario=='normal'or self.clock.t>=1006:row[2:4]=['COMPLETED','0:0'];code=0
        return dict(row=row,srun_exit=code)
    def cancel(self):self.canceled=True;return [dict(stage=s,exact_owned=True)for s in self.processes]


def loop_cases():
    rows=[]
    for scenario in ['normal','account_lag','stage_failure','spawn_failure','exit_unknown','cancel']:
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir();c=Clock();b=Recording(c,scenario)
            g=dict(run='cpu-r355-'+scenario,publication_commit='a'*40,owner_manifest_SHA256='b'*64,controller_manifest_SHA256='c'*64)
            def sleep(n):
                c.sleep(n)
                if scenario=='cancel':(root/'control'/('GLM-r355-cancel-'+g['run']+'.json')).write_text('{}')
            with patch.object(controller,'collect',return_value={'CPU_recording_only':True}):q=controller.execute(root,g,b,c.now,sleep)
            expected=scenario in ['normal','account_lag'];assert q['passed']==expected and q['completed']
            assert b.started==([0,1,2]if scenario=='spawn_failure'else[0,1,2,3])
            assert b.canceled!=expected
            if not expected:assert len(q['forced_cleanup'])==len(b.processes)and q['post_empty']
            rows.append(dict(scenario=scenario,expected_pass=expected,started=b.started,canceled=b.canceled,simulated_seconds=c.t-1000))
    return rows


def main():
    q=dict(completed=True,passed=True,GPU=0,Native_dlopen=False,real_Slurm=False,argv=argv_and_identity(),loop_cases=loop_cases())
    # Every call between the two independent forwards is local: no WORLD callback
    # occurs inside the retained354 helper's forward window.
    import ast
    t=ast.parse((D/'owner/owner_r290.py').read_bytes())
    functions={n.name:n for n in ast.walk(t)if isinstance(n,ast.FunctionDef)}
    for name in ['forward_values','dec_forward']:
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in ['control','barrier','cancel_vote']for n in ast.walk(functions[name]))
    q['no_WORLD_in_F0_F1_callbacks']=True
    (D/'controller-CPU-local.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({'passed':True,'GPU':0,'cases':len(q['loop_cases'])}))


if __name__=='__main__':main()
