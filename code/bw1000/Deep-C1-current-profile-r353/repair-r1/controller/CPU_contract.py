"""CPU source/resource/namespace contracts only; never invoke controller.main."""
import pathlib,json,sys,copy,hashlib,os
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import controller_348 as c
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
g=json.loads((D/'execution-GO.template.json').read_bytes());g.update(GPU_authorized=True,publication_commit='0'*40,controller_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))
c.verify_sources(g);cases=[]
for key,value in [('GPU_authorized',False),('cards',16),('TP',4),('max_HTTP',7),('max_Native',7),('ordinary_budget',129),('GPU_max_s',1801),('Native_max_s',1800),('CPU_keeper_max_s',2101),('GPU_node','a02r4n05'),('GPU_parent',910380),('cards',True),('owner_manifest_SHA256','0'*64),('CPU683_receipt_SHA256','0'*64)]:
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
q=dict(completed=True,passed=True,CPU_only=True,GPU=False,Native_dlopen=False,weights=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],owner_manifest_SHA256=c.OWNER_SHA,controller_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),CPU683_SHA256=c.CPU_SHA,actual_namespace_and_DSO_C621_inheritance=True,invalid_scope_and_accounting_cases=cases,original_bound_1800_includes_loading=True,source_budget={'GPU':1800,'Native_shell':1770,'worker':1680,'keeper':2100})
p=pathlib.Path('/refactor/candidates/Deep-C1-current-profile-r353r1-CPU/controller-contract.json');assert not p.exists();p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
