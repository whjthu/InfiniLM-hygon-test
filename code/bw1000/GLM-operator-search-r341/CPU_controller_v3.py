import pathlib,tempfile,json,runpy,sys,shutil,types,hashlib
from unittest.mock import patch
p=pathlib.Path(__file__).parent;source=p/'repair_v3';captured=[];records=[]
for scenario in ['normal','duplicate_PCI','incomplete','wrong_count','wrong_current']:
 with tempfile.TemporaryDirectory(prefix='GLM341-controller-') as tmp:
  home=pathlib.Path(tmp);r=home/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-operator-search-r341/repair_v3';inc.parent.mkdir(parents=True);shutil.copytree(source,inc);(r/'logs').mkdir();(r/'results').mkdir()
  probes=[{'node':node,'rank':i,'physical_PCI':str(i),'normal':True,'device_count':1,'current_device':0} for node,n in [('a02r3n16',7),('a02r4n07',4)] for i in range(n)]
  if scenario=='duplicate_PCI':probes[1]['physical_PCI']='0'
  if scenario=='incomplete':probes.pop()
  if scenario=='wrong_count':probes[1]['device_count']=2
  if scenario=='wrong_current':probes[1]['current_device']=1
  for i,q in enumerate(probes):(r/('results/GLM341d-map-CPU-rank'+str(i)+'.json')).write_text(json.dumps(q))
  def check(a,**kw):return a[5]+'|parent|RUNNING|0:0|none\n'
  def idle(a,**kw):return types.SimpleNamespace(returncode=0,stdout=json.dumps({'node':a[a.index('-w')+1],'kfd_pids':[]}),stderr='')
  def popen(a,**kw):
   mode='probe' if any('--job-name=GLM341d-probe-' in x for x in a) else 'measure';record=json.loads((r/('results/GLM341d-'+mode+'-launch.json')).read_text());assert len(record['GPU_submissions'])==len(captured)%2
   output=next(x.split('=',1)[1] for x in a if x.startswith('--output='));assert pathlib.Path(output).is_absolute() and pathlib.Path(kw['stdout'].name).is_absolute() and str(r) in output
   captured.append(a);return types.SimpleNamespace(pid=123)
  before=len(captured);failed=False
  with patch('pathlib.Path.home',return_value=home),patch('subprocess.check_output',side_effect=check),patch('subprocess.run',side_effect=idle),patch('subprocess.Popen',side_effect=popen):
   for mode in (['probe','measure'] if scenario=='normal' else ['measure']):
    with patch.object(sys,'argv',[str(inc/'controller.py'),'a'*40,mode]):
     try:runpy.run_path(str(inc/'controller.py'),run_name='__main__')
     except AssertionError:failed=True
  assert (not failed and len(captured)-before==4) if scenario=='normal' else (failed and len(captured)==before)
  records.append({'scenario':scenario,'expected_pass':scenario=='normal','observed_rejection':failed,'Popen_mocked_calls':len(captured)-before})
assert [int(a[a.index('-n7')][2:]) if '-n7' in a else 4 for a in captured]==[7,4,7,4]
out={'CPU_only':True,'GPU_initialized':False,'real_subprocesses_launched':0,'actual_controller_runpy':True,'source_manifest_SHA256':hashlib.sha256((source/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'cases':records,'captured_srun_argv':captured}
target=pathlib.Path(sys.argv[1]);assert not target.exists();target.write_text(json.dumps(out,indent=2));print('CONTROLLER_CPU_PASS')
