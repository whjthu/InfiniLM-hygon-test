import pathlib,json,hashlib,ast,importlib.util,types,contextlib,io,re
r=pathlib.Path(__file__).resolve().parent;d=r/'GLM-exact-budget-Model-private-r8855';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8855.json';m=json.loads(mp.read_text());receipt=json.loads((r/'GLM-exact-budget-CPU-observed-r8854-a.json').read_text(encoding='utf-8-sig'))['receipt']
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
spec=importlib.util.spec_from_file_location('actual_resident_control_8855',d/'glm_crossrequest_resident_base_r8855.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
assert base.BANK_BYTES==m['resident_physical_bytes_per_layer']==receipt['resident_exact_budget']['resident_bank_bytes']==167772160
assert base.affordable(40*base.BANK_BYTES+base.RESERVE+base.ALLOWANCE,1)==40 and base.affordable(40*base.BANK_BYTES+base.RESERVE+base.ALLOWANCE-1,1)==39
class FakeCuda:
 def synchronize(self,rank):pass
 def empty_cache(self):pass
 def mem_get_info(self,rank):return(14*1024**3,64*1024**3)
class FakeLib:
 def __init__(self,resetfail=False):self.resetfail=resetfail
 def modern_glm_idle_trim_r3771(self,ctx,a,n):
  for i,v in enumerate([0,0,0,0,1,2]):a[i]=v
  return 0
 def modern_glm_model_reset(self,ctx):return int(self.resetfail)
def actual_apply(setterfail=False,resetfail=False):
 c=base.ResidentControl.__new__(base.ResidentControl);c.torch=types.SimpleNamespace(cuda=FakeCuda());c.lib=FakeLib(resetfail);c.ctx=None;c.stage=1;c.rank=0;c.epoch=0;c.chosen=None;c.report={};c.runtime=99;c.save=lambda:None;c.calls=[]
 def check(code):
  if code:raise RuntimeError('existing_model_error_getter: '+('cleanup reset failed'if resetfail else'setter rejected'))
 c.check=check
 def setter(bits,trace):c.calls.append(sum(bits));return int(setterfail and any(bits))
 c.setter=setter
 def allrows(values):
  if len(values)==5:return[[14*1024,values[1],values[2],values[3],values[4]]for _ in range(16)]
  assert len(values)==4
  return[[int(not(setterfail and s==1)),38 if s==0 else 40,values[2],values[3]]for s in(0,1)for _ in range(8)]
 c.all_rows=allrows;c.vote=lambda ok:bool(ok)
 c.snapshot=lambda:{'resident_physical_bytes_per_layer':base.BANK_BYTES,'Context_mask':list(map(bool,base.make_mask(1,40))),'Runtime_identity':99,'memory_trace':False,'any_enabled':True,'layers':[{'global_layer':38+i,'enabled':True,'active_rows':0,'key_shape':[32,640,8,256],'value_shape':[32,640,8,256]}for i in range(40)]}
 try:
  with contextlib.redirect_stdout(io.StringIO()):base.ResidentControl.apply(c,'actual-CPU-branch',True,False)
 except Exception as e:return c,e
 return c,None
c,e=actual_apply();assert e is None and c.report['resident_policy_phases'][0]['physical_budget_bytes']==40*base.BANK_BYTES
c,e=actual_apply(setterfail=True);assert isinstance(e,base.CandidateRejected)and c.report['resident_setter_rejection']['local_detail'].startswith('existing_model_error_getter:')and c.calls==[0,40,0]
c,e=actual_apply(setterfail=True,resetfail=True);assert isinstance(e,RuntimeError)and not isinstance(e,base.CandidateRejected)and 'resident_policy_phases'not in c.report and 'resident_setter_rejection'not in c.report
tree=ast.parse((d/'audit_GLM_prefill_Model_r8855.py').read_bytes());branch=next(x for x in ast.walk(tree)if isinstance(x,ast.If)and ast.unparse(x.test)=='phase_after is None');code=compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'actual-auditor-missing-phaseafter','exec')
for q,cases,p in[({},[1],[{}]),({'resident_candidate_rejected':True,'resident_setter_rejection':{'statuses':[]}},[],None),({'resident_candidate_rejected':True,'resident_admission_rejection':{'phase':'x'}},[],None)]:
 env={'q':q,'cases':cases,'phase_after':p};exec(code,env)
for q,cases in[({},[]),({'resident_candidate_rejected':True},[]),({'resident_candidate_rejected':True,'resident_setter_rejection':{}},[1])]:
 try:exec(code,{'q':q,'cases':cases,'phase_after':None})
 except AssertionError:pass
 else:raise AssertionError('unknown missingphaseafter accepted')
worker=ast.parse((d/'modern_glm_prefill_micro_r8855.py').read_bytes());pred=next(x for x in ast.walk(worker)if isinstance(x,ast.Assert)and "report['cache_after_destroy']['active']"in ast.unparse(x));code=compile(ast.fix_missing_locations(ast.Module(body=[pred],type_ignores=[])),'actual-cleanup-cache-zero','exec')
exec(code,{'report':{'cache_after_destroy':{'created':1,'destroyed':1,'active':0,'active_bytes':0}}})
try:exec(code,{'report':{'cache_after_destroy':{'created':1,'destroyed':0,'active':1,'active_bytes':160}}})
except AssertionError:pass
else:raise AssertionError('cleanup failure accepted')
ns={};exec((d/'glm_crossrequest_contract_r8855.py').read_bytes(),ns);g=json.loads((d/'GLM477-frozen-gold-r8844.json').read_bytes());prompts=g['input_ids']
for n in(16,32):
 ids=g['generated_ids'][:n];eos=g['stopped_at_EOS'][:n];assert ns['validate_reference']('I512','R',prompts[:n],ids,eos,128,{},prompts,{},g)[0];bad=[list(x)for x in ids];bad[-1][0]=(bad[-1][0]+1)%154880;assert not ns['validate_reference']('I512','R',prompts[:n],bad,eos,128,{},prompts,{},g)[0]
linked=json.loads((r/'GLM-exact-budget-callchain-CPU-r8854-b.json').read_bytes());chain=linked['actual_linked_callchain'][0];ins='\n'.join(chain['rows32_branch_and_budget']);assert 'cmp    $0x20,%rax'in ins and 'lea    (%rdx,%rdx,4),%rdx'in ins and 'shl    $0x19,%rdx'in ins
symbols=set()
for n in['modern_glm_prefill_micro_r8855.py','glm_prefill_micro_probe_r8855.py','glm_crossrequest_control_r8855.py','glm_crossrequest_resident_base_r8855.py','glm_layer77_policy_r8855.py','glm_window_control_r8855.py']:symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',(d/n).read_text()))
original=(d/'modern_glm_full_model_probe_r2.py').read_text().split('def worker(rank,payload):')[0];symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',original))
out={'CPU_only':True,'GPU':False,'dlopen':False,'passed':True,'manifest_SHA256':sha(mp),'entry_SHA256':sha(d/'dispatch_GLM_resident_exact_Model_r8855.py'),'DSO_SHA256':receipt['library_SHA256'],'actual_ResidentControl_apply_success_setterrejection_cleanupfailure':'PASS','actual_auditor_missing_phase_after_known_rejection_unknownnegative':'PASS','actual_worker_cleanup_cachezero_positive_negative':'PASS','original477_B16_B32_gold_positive_badtokennegative':'PASS','Python_manifest_actual_CPUbudget_bytes':base.BANK_BYTES,'linked_actual_setter_budget':'ROWS32 cmp32 branch:lea rdx+rdx*4 then shl25 =count*167772160; actualDSO not only standalonechecker','linked_disassembly_SHA256':chain['SHA256'],'actual_consumer_CAPI_names':sorted(symbols),'budget':'45minhard2600inner new8855;1549sactual558firstpolicy+B16~146+B32~300+QA/reset/shutdownmargin;noextraarm/retry/extension'}
(r/'GLM-exact-budget-Model-CPUchecks-r8855.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
