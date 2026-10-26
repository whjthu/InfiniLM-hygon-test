import ast,pathlib,types,json
src=ast.parse(pathlib.Path(__file__).with_name('target_model.py').read_text());branch=next(n for n in ast.walk(src)if isinstance(n,ast.If)and ast.unparse(n.test)=="report['capacity_only_mode']");final=next(n for n in src.body if isinstance(n,ast.Try)).finalbody
rows=[]
for only in [True,False]:
 calls={'load':0,'forward':0,'destroy':0,'control':[]};report={'capacity_only_mode':only}
 class Control:
  def collect(self,k,v):calls['control'].append(k);return[]
  def set_timeout(self,v):pass
  def close(self):pass
 class Lib:
  def modern_kimi_engine_keys_r2475(self,c):return b'["dummy"]'
  def modern_kimi_engine_output_metrics_r2475(self,c):return b'{}'
  def modern_kimi_engine_destroy_r2475(self,c):calls['destroy']+=1;return 0
 def load(*a,**k):calls['load']+=1;return {'loaded_keys':1}
 def run(g):calls['forward']+=1;report['serving_protocol_complete']=True
 env={'keys':{'dummy'},'report':report,'control':Control(),'lib':Lib(),'ctx':1,'sampler':None,'model':None,'text':{},'safe_open':None,'torch':None,'check':lambda x:None,'save':lambda:None,'load_parameters':load,'memory':lambda:[],'memory_guard':lambda x:{'minimum_free_bytes':5*1024**3},'verify_maps':lambda x:{},'manifest':{},'library':'fake','sha':lambda p:'fake','target_protocol':types.SimpleNamespace(run=run),'verify_owner_affinity':lambda x:{},'json':json}
 exec(compile(ast.Module([branch],type_ignores=[]),'actualcapacitybranch','exec'),env);exec(compile(ast.Module(final,type_ignores=[]),'actualfinally','exec'),env);assert calls['load']==int(not only)and calls['forward']==int(not only)and calls['destroy']==1 and report['completed']and report['all8_owned_workers_shutdown_completed'];rows.append({'capacity_only':only,'calls':calls,'completed':report['completed'],'actual_source_AST_controlflow':True})
q={'CPU_only':True,'GPU':False,'actual_capacity_branch_and_finally_covered':rows,'no_real_Model_runtime_claim':True};pathlib.Path(__file__).with_name('CPU-capacity-mode-r9145.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
