"""Actual controller.main with only Slurm/process leaves faked; no GPU submit."""
import pathlib,json,sys,copy,hashlib,os,tempfile,types
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import controller_366 as c
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
g=json.loads((D/'execution-GO.template.json').read_bytes());g.update(GPU_authorized=True,publication_commit='0'*40,controller_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))
c.verify_sources(g);cases=[]
for key,value in [('GPU_authorized',False),('cards',16),('TP',4),('max_HTTP',7),('max_Native',7),('ordinary_budget',129),('GPU_max_s',2701),('Native_max_s',2700),('CPU_keeper_max_s',3001),('GPU_node','a02r4n05'),('GPU_parent',910380),('cards',True),('owner_manifest_SHA256','0'*64),('CPU_entry_receipt_SHA256','0'*64)]:
 bad=copy.deepcopy(g);bad[key]=value
 try:c.validate_scope(bad)
 except(AssertionError,PermissionError):cases.append([key,str(value),'rejected'])
 else:raise AssertionError('invalid scope accepted '+key)
good=['910374.999',c.LABEL,'COMPLETED','0:0','a02r4n01'];c.validate_account_row(good)
for index,value in [(0,'910380.999'),(1,'foreign-label'),(4,'a02r4n05')]:
 row=good[:];row[index]=value
 try:c.validate_account_row(row)
 except AssertionError:cases.append(['account-row-'+str(index),value,'rejected'])
 else:raise AssertionError('foreign accounting row accepted')
main_cases=[]
for state in ['COMPLETED','FAILED']:
 with tempfile.TemporaryDirectory(prefix='Deep366-controller-CPU-')as td:
  root=pathlib.Path(td);(root/'results').mkdir();(root/'logs').mkdir();go=root/'go.json';go.write_text(json.dumps(g));old=(c.R,c.W,c.verify_sources,c.subprocess.check_output,c.subprocess.run,c.subprocess.Popen,c.time.sleep,sys.argv);c.R=root;c.W=root/'watch.json';c.verify_sources=lambda actual: c.validate_scope(actual);submitted=[]
  def output(args,**kwargs):
   if args[:3]==['scontrol','show','job']:return 'UserId=aclsylqidf(123) JobState=RUNNING NodeList='+('a14r2n20'if args[3]=='910407'else c.NODE)
   if args[:3]==['scontrol','show','hostnames']:return args[3]+'\n'
   assert args[0]=='sacct'
   return ('910374.999|'+c.LABEL+'|'+state+'|'+('0:0'if state=='COMPLETED'else'1:0')+'|'+c.NODE+'\n')if submitted else ''
  def run(args,**kwargs):
   assert args[0]=='srun'and'--gres=none'in args
   return types.SimpleNamespace(returncode=0,stdout=json.dumps({'node':c.NODE,'kfd_pids':[],'HTTP43882_bindable':True,'actual_step':'910374.998'})+'\n')
  def popen(args,**kwargs):
   assert args[0]=='srun'and'--gres=dcu:8'in args and'00:45:00'in args and'-c32'in args and len(submitted)==0;submitted.append(args)
   report=dict(completed=True,passed=True,normal_server_shutdown=True,normal_Native_shutdown=True,actual_HTTP=6,actual_Native=6,owner_exit_observation={'all8_unique_threads_exited':True},CPU_FAKE_MODEL_REPORT=True)
   (root/'results'/('Deep-HC-T1-Model-r366-910374.999.json')).write_text(json.dumps(report));return types.SimpleNamespace(pid=123,poll=lambda:0)
  c.subprocess.check_output=output;c.subprocess.run=run;c.subprocess.Popen=popen;c.time.sleep=lambda _:None;sys.argv=['controller_366.py','--execution-go',str(go)]
  try:c.main();watch=json.loads(c.W.read_bytes());assert watch['completed']and watch['post_empty']and watch['actual_GPU']=='910374.999'and watch['Model_qualified']==(state=='COMPLETED')and len(submitted)==1
  finally:c.R,c.W,c.verify_sources,c.subprocess.check_output,c.subprocess.run,c.subprocess.Popen,c.time.sleep,sys.argv=old
  main_cases.append(dict(kind='actual_controller_main',Slurm_terminal=state,passed=True,one_submit_leaf_faked=True,fresh_idle_and_post_leaf_faked=True,Model=False,GPU=False))
q=dict(completed=True,passed=True,CPU_only=True,GPU=False,Native_dlopen=False,weights=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],owner_manifest_SHA256=c.OWNER_SHA,controller_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),CPU_entry_SHA256=c.CPU_SHA,actual_namespace_and_DSO_C621_inheritance=True,invalid_scope_and_accounting_cases=cases,actual_main_cases=main_cases,original_bound_2700_includes_loading=True,source_budget={'GPU':2700,'Native_shell':2670,'worker':2580,'keeper':3000})
p=pathlib.Path('/refactor/candidates/Deep-HC-T1-Model-r366-owner-CPU-v3/controller-contract.json');assert not p.exists();p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
