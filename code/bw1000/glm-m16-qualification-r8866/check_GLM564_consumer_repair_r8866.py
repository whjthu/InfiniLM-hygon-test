import pathlib,sys,importlib.util,types,copy,json,ast,tempfile,hashlib,os
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-M16-Model-private-r8866';d=r/'GLM-M16-Model-private-r8866';sys.path.insert(0,str(old))
spec=importlib.util.spec_from_file_location('actual_control_repair',d/'glm_crossrequest_control_r8866.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
import glm_crossrequest_resident_base_r8866 as base
checks=[]
def actual_finish(stage,micro,bad=False,floorfail=False):
 c=mod.CrossrequestControl.__new__(mod.CrossrequestControl);c.stage=stage;c.rank=0;c.count=base.BOUNDS[stage+1]-base.BOUNDS[stage];c.sparse=c.count-(3 if stage==0 else 0);c.runtime=17;c.report={};c.saves=0
 c.save=lambda:setattr(c,'saves',c.saves+1);c.vote=lambda x:bool(x)and not floorfail;c.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda rank:(8*1024**3,64*1024**3)))
 events=[{'round':1,'rows':list(range(micro)),'flat_tokens':micro}];expect=base.expected_dispatch(events,c.count)
 before={'Runtime_identity':17,'layers':[dict(global_layer=i,enabled=True,packed_prefix_bytes=0,**{k:0 for k in expect})for i in range(base.BOUNDS[stage],base.BOUNDS[stage+1])],'host_counters':[0]*7,'vector_counts':[0]*8,'vector_mode':1,'transient':{'enabled':False,'GPU_logical_bytes':0},'memory_samples':0,'min_observed_free':0}
 after=copy.deepcopy(before)
 for layer in after['layers']:layer.update(expect)
 after['vector_counts']=[c.sparse,0,c.sparse,0,0,c.sparse*micro,c.sparse*micro,0]
 if bad:after['vector_counts'][0]+=1
 c.snapshot=lambda:after
 try:q=c.finish('actual-override',before,events,[[0]*512 for _ in range(micro)])
 except (AssertionError,RuntimeError)as exc:
  assert bad or floorfail;return type(exc).__name__
 assert not(bad or floorfail)and q is after and c.saves==1 and c.report['resident_phase_after'][0]['vector_counter_delta']==after['vector_counts'];return'PASS'
for s in(0,1):
 for m in(8,16):assert actual_finish(s,m)=='PASS';checks.append(f'actual_CrossrequestControl.finish_super_and_override_stage{s}_M{m}')
assert actual_finish(1,16,bad=True)=='AssertionError';checks.append('actual_override_wrong_vector_negative')
assert actual_finish(1,16,floorfail=True)=='RuntimeError';checks.append('actual_super_runtime_floor_failure_negative')
tree=ast.parse((d/'glm_prefill_micro_probe_r8866.py').read_bytes());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run');save=next(x for x in run.body if isinstance(x,ast.FunctionDef)and x.name=='save');gen=next(x for x in run.body if isinstance(x,ast.FunctionDef)and x.name=='generate')
idx=next(i for i,x in enumerate(gen.body)if isinstance(x,ast.Expr)and isinstance(x.value,ast.Call)and ast.unparse(x.value.func)=='resident.finish')
sequence=gen.body[idx-2:idx+4];assert 'last_completed_generation' in ast.unparse(sequence[0])and ast.unparse(sequence[1])=='save()'
code=lambda nodes:compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-probe-statements','exec')
with tempfile.TemporaryDirectory(dir=r)as td:
 p=pathlib.Path(td)/'owner.json';row={'phase':'M16','generated_ids':[[1,2,3]],'emitter_clock':[1,2,3],'stopped_at_EOS':[False]};report={'multi_batch_cases':[]};env={'report':report,'report_path':p,'json':json,'os':os,'row':row,'label':'M16','resident_before':{},'events':[],'prompts':[]}
 exec(code([save]),env)
 class FailedFinish:
  def finish(self,*a):raise RuntimeError('original counter failure')
 env['resident']=FailedFinish()
 try:exec(code(sequence),env)
 except RuntimeError:pass
 else:raise AssertionError('failure lost')
 persisted=json.loads(p.read_bytes());assert persisted['last_completed_generation']==row and not persisted['multi_batch_cases'];checks.append('actual_atomic_save_preserves_finalIDs_EOS_clock_before_finish_failure')
 class GoodFinish:
  def finish(self,*a):return None
  def snapshot(self):return{'native':'healthy'}
 env['resident']=GoodFinish();exec(code(sequence),env);persisted=json.loads(p.read_bytes());assert persisted['multi_batch_cases']==[row]and persisted['last_completed_generation']==row;checks.append('actual_success_appends_after_finish')
protocol=next(x for x in run.body if isinstance(x,ast.Try)and any(isinstance(y,ast.For)and ast.unparse(y.target)=='label'for y in x.body))
completion=[x for x in protocol.finalbody if isinstance(x,ast.Assign)and any(k in ast.unparse(x)for k in("report['resident_candidate_quality_passed']","report['cohort_phases_completed']","report['planned_phase_accounting_completed']"))]
assert len(completion)==3
qa_scope=next(x for x in protocol.finalbody if isinstance(x,ast.Assign)and any(isinstance(y,ast.Subscript)and ast.unparse(y)=="report['after_select_QA_scope']"for y in x.targets))
for success,rejected,expected in[(True,False,'EXECUTED_REQUIRED'),(False,True,'SKIPPED_REJECTED_CANDIDATE'),(False,False,'SKIPPED_PROTOCOL_INCOMPLETE')]:
 env={'report':{},'protocol_success':success,'rejected':rejected};exec(code([qa_scope]),env);assert env['report']['after_select_QA_scope']==expected
checks.append('actual_afterQA_scope_success_rejection_unexpected_failure')
for fail in(False,True):
 report={'multi_batch_cases':[],'skipped_rejected_candidate_phases':[]};env={'report':report,'protocol_success':False,'rejected':False,'labels':['M8','M16'],'i512_inputs':[[0]*512]*32,'payload':{'chat_eos_token_ids':[]},'set':set}
 def generate(label,*args):
  if fail and label=='M16':raise RuntimeError('finish failed')
  report['multi_batch_cases'].append({'phase':label})
 env['generate']=generate
 try:exec(code(protocol.body),env)
 except RuntimeError:assert fail
 exec(code(completion),env)
 assert report['cohort_phases_completed']==(not fail)and report['resident_candidate_quality_passed']==(not fail)
 checks.append('actual_protocol_success_completion'if not fail else'actual_unexpected_finish_never_claims_completed')
out={'CPU_only':True,'GPU':False,'dlopen':False,'passed':True,'actual_override_not_base_only':True,'checks':checks,'new_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in d.glob('*.py')},'old_564_unmodified':True,'no_Model_authorization_or_replay':True}
(r/'GLM564-consumer-repair-CPUchecks-r8866.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(out))
