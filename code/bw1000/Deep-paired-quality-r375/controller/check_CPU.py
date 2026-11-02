"""Recording Slurm leaves drive actual per-replica controller.main; no GPU."""
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace as NS

D = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(D))
cases = []


def load(replica):
    os.environ['DEEP_R375_REPLICA'] = str(replica)
    spec = importlib.util.spec_from_file_location('check375r' + str(replica), D / 'controller_375.py')
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    return c


def scope(c):
    return dict(GPU_authorized=True, publication_branch='bw1000-experiments', publication_commit='1'*40,
                replica=c.REPLICA, GPU_parent=c.PARENT, GPU_node=c.NODE,
                CPU_keeper_parent=910407, CPU_keeper_node='a14r2n20', cards=8, TP=8, PP=1,
                context=32768, ordinary_budget=4096, thinking_budget=8192, max_HTTP=66, max_Native=66,
                GPU_max_s=7200, Native_max_s=7140, CPU_keeper_max_s=7800, prefill_chunk=1024,
                one_attempt=True, no_retry_reload=True, owner_manifest_SHA256=c.OWNER_SHA,
                controller_manifest_SHA256='2'*64, CPU_source_manifest_SHA256=c.CPU_SOURCE_SHA,
                CPU_actual_step='910407.751', CPU_entry_receipt_SHA256='3'*64,
                wrapper_SHA256=c.NAMESPACE_WRAPPER_SHA, DSO_CPU621_receipt_SHA256=c.CPU621_SHA,
                Native_SHA256='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1',
                encoding_receipt_SHA256='f17342d2ce78e73b9f8853720c6b477c947ccf8d71a7503dcd7e04e391e16447',
                encoding_receipt_container_path='/refactor/candidates/Deep-paired-quality-r375/CPU-910407.749/encoded/encoding.json')


for replica in range(4):
    c=load(replica);g=scope(c);c.validate_scope(g)
    assert (c.PARENT,c.NODE)==[(910374,'a02r4n01'),(910374,'a02r4n05'),(910374,'a02r4n07'),(910371,'a02r3n16')][replica]
    for key, value in [('replica',(replica+1)%4),('GPU_parent',910380),('GPU_node','a02r4n04'),('cards',True),
                       ('TP',4),('max_HTTP',65),('max_Native',67),('ordinary_budget',128),
                       ('GPU_max_s',7201),('Native_max_s',7200),('CPU_keeper_max_s',7801),('GPU_authorized',False)]:
        bad=copy.deepcopy(g);bad[key]=value
        try:c.validate_scope(bad)
        except (AssertionError,PermissionError):pass
        else:raise AssertionError('Bad scope accepted '+key)
    good=[str(c.PARENT)+'.999',c.LABEL,'COMPLETED','0:0',c.NODE];c.validate_account_row(good)
    for i,value in [(0,'910380.999'),(1,'foreign-label'),(4,'a02r4n04')]:
        bad=good.copy();bad[i]=value
        try:c.validate_account_row(bad)
        except AssertionError:pass
        else:raise AssertionError('Foreign account row accepted')
    cases.append(dict(kind='exact_replica_scope_and_accounting',replica=replica,passed=True))
    for scenario in ['success','Slurm-failure','srun-nonzero','srun-lag','existing-watch','existing-label','foreign-node-step','bad-final-report','report-timeout-still-post']:
        with tempfile.TemporaryDirectory(prefix='Deep375-controller-') as td:
            root=pathlib.Path(td);(root/'results').mkdir();(root/'logs').mkdir();go=root/'go.json';go.write_text(json.dumps(g))
            prior=(c.R,c.W,c.verify_sources,c.subprocess.check_output,c.subprocess.run,c.subprocess.Popen,c.time,c.socket.gethostname,sys.argv,os.environ.get('SLURM_JOB_ID'),os.environ.get('SLURM_STEP_ID'))
            c.R=root;c.W=root/'watch.json';c.verify_sources=lambda actual:c.validate_scope(actual)
            os.environ['SLURM_JOB_ID']='910407';os.environ['SLURM_STEP_ID']='999';c.socket.gethostname=lambda:'a14r2n20'
            submitted=[];clock=[1000.0];polls=[0]
            c.time=NS(time=lambda:clock[0]+100000,monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
            if scenario=='existing-watch':c.W.write_text('')
            def output(argv,**kw):
                if argv[:3]==['scontrol','show','job']:
                    return 'UserId=aclsylqidf(123) JobState=RUNNING NodeList='+('a14r2n20' if argv[3]=='910407' else c.NODE)
                if argv[:3]==['scontrol','show','hostnames']:return argv[3]+'\n'
                assert argv[0]=='sacct' and argv[argv.index('-j')+1]==str(c.PARENT)
                if scenario=='existing-label' and not submitted:return str(c.PARENT)+'.998|'+c.LABEL+'|RUNNING|0:0|'+c.NODE+'\n'
                if scenario=='foreign-node-step' and not submitted:return str(c.PARENT)+'.998|foreign|RUNNING|0:0|'+c.NODE+'\n'
                if not submitted:return ''
                state,code=('FAILED','1:0')if scenario=='Slurm-failure'else('COMPLETED','0:0')
                return str(c.PARENT)+'.999|'+c.LABEL+'|'+state+'|'+code+'|'+c.NODE+'\n'
            def run(argv,**kw):
                assert argv[0]=='srun' and '--gres=none' in argv
                assert '--jobid='+str(c.PARENT) in argv and argv[argv.index('-w')+1]==c.NODE
                assert '--node' in argv and argv[argv.index('--node')+1]==c.NODE
                tids=json.loads(argv[argv.index('--owner-tids')+1])
                return NS(returncode=0,stdout=json.dumps(dict(node=c.NODE,kfd_pids=[],HTTP43882_bindable=True,
                        actual_step=str(c.PARENT)+'.997',owner_tids=tids,owner_tids_all_gone=True))+'\n')
            def popen(argv,**kw):
                assert len(submitted)==0 and '--gres=dcu:8' in argv and '-c32' in argv and '02:00:00' in argv
                assert '--jobid='+str(c.PARENT) in argv and argv[argv.index('-w')+1]==c.NODE
                assert 'DEEP_R375_REPLICA='+str(replica) in argv[argv.index(next(x for x in argv if x.startswith('--export='))) ]
                exported=next(x for x in argv if x.startswith('--export='))[len('--export=ALL,'):]
                for item in exported.split(','):
                    key,value=item.split('=',1);assert kw['env'][key]==value
                assert kw['env']['DEEP_R375_REPLICA']==str(replica)
                assert kw['start_new_session'] is True;submitted.append(argv)
                report=dict(completed=True,passed=scenario!='bad-final-report',normal_server_shutdown=True,
                            normal_Native_shutdown=True,actual_HTTP=66,actual_Native=66,attempted_Native=66,completed_Native=66,
                            owner_exit_observation={'all8_unique_threads_exited':True,'owners':[{'tid':99000+i}for i in range(8)]},
                            CPU_RECORDING_NOT_MODEL=True)
                if scenario!='report-timeout-still-post':
                    (root/'results'/('Deep-paired-quality-r375-'+str(c.PARENT)+'.999.json')).write_text(json.dumps(report))
                def poll():
                    polls[0]+=1
                    if scenario=='srun-lag'and polls[0]<3:return None
                    return 1 if scenario in ['Slurm-failure','srun-nonzero']else 0
                return NS(pid=123,poll=poll)
            c.subprocess.check_output=output;c.subprocess.run=run;c.subprocess.Popen=popen;sys.argv=['controller_375.py','--execution-go',str(go)]
            error=None
            try:
                try:c.main()
                except BaseException as exc:error=repr(exc)
                if scenario in ['existing-watch','existing-label','foreign-node-step']:
                    assert error is not None and not submitted
                elif scenario=='report-timeout-still-post':
                    watch=json.loads(c.W.read_text());assert error is not None and watch['completed'] and watch['post_empty'] and len(submitted)==1
                    assert watch['post_owner_TID_count']==0 and watch['post_owner_TID_evidence']=='UNKNOWN_NO_THREAD_PROOF'
                    assert 'terminal_report_error'in watch and not watch['Model_qualified']
                else:
                    watch=json.loads(c.W.read_text());assert error is None,error
                    assert watch['completed'] and watch['post_empty'] and len(submitted)==1
                    assert watch['actual_GPU']==str(c.PARENT)+'.999'
                    assert watch['Model_qualified']==(scenario in ['success','srun-lag'])
                    if scenario in ['success','srun-lag','bad-final-report']:assert len(watch['post']['owner_tids'])==8
                cases.append(dict(kind='actual_controller_main_recording_leaves',replica=replica,scenario=scenario,passed=True,
                                  GPU_submits_recorded=len(submitted),actual_GPU=False,error=error))
            finally:
                c.R,c.W,c.verify_sources,c.subprocess.check_output,c.subprocess.run,c.subprocess.Popen,c.time,c.socket.gethostname,sys.argv,oldjob,oldstep=prior
                if oldjob is None:os.environ.pop('SLURM_JOB_ID',None)
                else:os.environ['SLURM_JOB_ID']=oldjob
                if oldstep is None:os.environ.pop('SLURM_STEP_ID',None)
                else:os.environ['SLURM_STEP_ID']=oldstep
    # No wall sleep for NFS artifact visibility/Native-finally propagation.
    for name,states,success,expected_reads in [
        ('missing-then-complete',[None,{'completed':True,'passed':True,'normal_Native_shutdown':True}],True,2),
        ('partial-then-complete',['{',{'completed':True,'passed':True,'normal_Native_shutdown':True}],True,2),
        ('stale-false-then-complete',[{'completed':False},{'completed':True,'passed':True,'normal_Native_shutdown':True}],True,2),
        ('pending-native-finally',[{'completed':True,'passed':True,'normal_Native_shutdown':False},{'completed':True,'passed':True,'normal_Native_shutdown':True}],True,2),
        ('explicit-final-failure',[{'completed':True,'passed':False,'normal_Native_shutdown':False}],True,1),
        ('forever-missing',[None],False,None)]:
        oldtime=c.time;tick=[0];reads=[0]
        c.time=NS(monotonic=lambda:tick[0],sleep=lambda s:tick.__setitem__(0,tick[0]+s))
        class Path:
            def read_bytes(self):
                value=states[min(reads[0],len(states)-1)];reads[0]+=1
                if value is None:raise FileNotFoundError()
                return value.encode() if isinstance(value,str)else json.dumps(value).encode()
        try:
            result=None;error=None
            try:result=c.terminal_report(Path(),visibility_s=3)
            except TimeoutError as exc:error=exc
            assert (error is None)==success
            if expected_reads is not None:assert reads[0]==expected_reads
            else:assert tick[0]==3
        finally:c.time=oldtime
        cases.append(dict(kind='terminal_artifact_visibility',replica=replica,scenario=name,passed=True,reads=reads[0]))

# Actual aggregate launcher argv/file handoff; only source/parent/Popen leaves recorded.
spec=importlib.util.spec_from_file_location('check375_all',D/'invoke_all_375.py')
all_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(all_module)
for scenario in ['four-keepers','wrong-node-suppresses-all']:
    with tempfile.TemporaryDirectory(prefix='Deep375-all-CPU-')as td:
        root=pathlib.Path(td);(root/'results').mkdir();(root/'logs').mkdir();modules=[load(i)for i in range(4)]
        go=dict(GPU_authorized=True,one_attempt=True,publication_commit='1'*40,replicas=[scope(c)for c in modules])
        if scenario=='wrong-node-suppresses-all':go['replicas'][2]['GPU_node']='a02r4n04'
        path=root/'set.json';path.write_text(json.dumps(go));submitted=[];parents=[]
        for c in modules:
            c.W=root/(c.LABEL+'-watch.json')
            c.verify_sources=lambda g,c=c:c.validate_scope(g)
            c.parent=lambda parent,node,env:parents.append((parent,node))
        old=(all_module.R,all_module.load,all_module.subprocess.Popen,sys.argv)
        all_module.R=root;all_module.load=lambda i:modules[i]
        def popen(argv,**kw):
            replica=len(submitted);assert '--gres=none'in argv and '-c1'in argv and '--mem=4096M'in argv and '02:10:00'in argv
            assert '--jobid=910407'in argv and argv[argv.index('-w')+1]=='a14r2n20'
            assert '--export=ALL,DEEP_R375_REPLICA='+str(replica)in argv and kw['start_new_session']is True
            assert kw['env']['DEEP_R375_REPLICA']==str(replica)
            submitted.append(argv);return NS(pid=8000+replica)
        all_module.subprocess.Popen=popen;sys.argv=['invoke_all_375.py','--execution-go-set',str(path)]
        try:
            error=None
            try:all_module.main()
            except AssertionError as exc:error=exc
            if scenario=='four-keepers':
                assert error is None and len(submitted)==4 and len(parents)==8
                record=json.loads((root/'results/Deep-paired-quality-r375-all-keeper-launch.json').read_bytes());assert len(record['submissions'])==4
                assert [x['GPU_node']for x in record['submissions']]==['a02r4n01','a02r4n05','a02r4n07','a02r3n16']
                try:all_module.main()
                except FileExistsError:pass
                else:raise AssertionError('Duplicate aggregate invocation did not stop')
                assert len(submitted)==4
            else:assert error is not None and not submitted
        finally:all_module.R,all_module.load,all_module.subprocess.Popen,sys.argv=old
        cases.append(dict(kind='actual_invoke_all_main_recording_leaves',scenario=scenario,passed=True,CPU_keeper_submits_recorded=len(submitted),actual_GPU=False))

receipt=dict(completed=True,passed=True,CPU_only=True,GPU=False,Model=False,Native_dlopen=False,
             scope='Actual controller.main/validate_scope with recording Slurm/process/idle leaves; no actual Context or GPU',
             controller_SHA256=hashlib.sha256((D/'controller_375.py').read_bytes()).hexdigest(),cases=cases)
print(json.dumps(receipt))
