import ast,pathlib,json,hashlib,tempfile,datetime,types
r=pathlib.Path('.');proof={};ns={'hashlib':hashlib,'json':json,'datetime':datetime}
def defs(name,names):
 t=ast.parse((r/name).read_text());v=[x for x in t.body if isinstance(x,ast.FunctionDef)and x.name in names];assert len(v)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=v,type_ignores=[])),name,'exec'),ns)
defs('launch_Kimi_TP16PP2_I192_r8669zh.py',['owned_native_failfast_r8669zh'])
with tempfile.TemporaryDirectory()as folder:
 root=pathlib.Path(folder);(root/'logs').mkdir();run='TP16PP2-I1928669zh-CPUtest';job='Kimi-TP16PP2-I192-'+run;row={'JobID':'910375.999999','JobName':job,'State':'RUNNING'}
 watch={'run':run,'actual_handle':row['JobID']};events=[]
 def signal(args,**kw):events.append(('signal',args));return types.SimpleNamespace(returncode=0,stdout='',stderr='')
 def save():events.append(('save',dict(watch.get('owner_error_failfast',{}))))
 (root/'logs'/f'othernonce-{row["JobID"]}-0.log').write_text('exception during forward: I192launchcode=1')
 own=root/'logs'/f'{job}-{row["JobID"]}-0.log';own.write_text('normal forward and warning')
 assert not ns['owned_native_failfast_r8669zh'](root,row,job,run,watch,save,signal)and not events
 own.write_text('normal\nexception during forward: I192launchcode=1\n')
 assert ns['owned_native_failfast_r8669zh'](root,row,job,run,watch,save,signal)
 assert [v[0]for v in events]==['save','signal','save']and events[1][1]==['scancel','--signal=TERM','910375.999999']
 assert events[0][1]['signal_attempted']and watch['owner_error_failfast']['signal_returncode']==0
 assert not ns['owned_native_failfast_r8669zh'](root,row,job,run,watch,save,signal)and len(events)==3
 bad={'run':run,'actual_handle':'910375.999998'}
 try:ns['owned_native_failfast_r8669zh'](root,row,job,run,bad,save,signal);raise RuntimeError('wronghandle accepted')
 except AssertionError:pass
 watch={'run':run,'actual_handle':row['JobID']}
 def broken(*a,**k):raise TimeoutError('CPU injection')
 assert ns['owned_native_failfast_r8669zh'](root,row,job,run,watch,save,broken)and 'signal_error'in watch['owner_error_failfast']
 assert not ns['owned_native_failfast_r8669zh'](root,row,job,run,watch,save,broken)
proof['actual_failfast_exactnonce_handle_no_warning_broadcancel_or_repeat_persist_before_oneTERM']=True
proof['signal_exception_preserved_no_retry_same_originalhandle']=True
defs('kimi_TP16PP2_I192_protocol_r8669zh.py',['frozen648_reference_r8669zh'])
expected=[[i+j for j in range(128)]for i in range(16)];q={'TP':16,'PP':2,'node':2,'PP_stage':1,'cache_geometry':dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024),'serving_cases':[{'label':'TP16-B16M16-O-full-reference128','B':16,'frontend_cohort_M':16,'policy':1,'I192_candidate_ON':False,'timed_decode_token_only':False,'input_lengths':[512]*16,'generated_ids':expected}]};raw=json.dumps(q).encode()
class FakePath:
 def __truediv__(self,x):return self
 def read_bytes(self):return raw
 def __str__(self):return 'CPU_schema_fixture_not_GPU_gold'
class Hash:
 def hexdigest(self):return '5c4c52c8895858be6eed7d5d48a60d2499c0d779c170cf55d11b7d64e11b91f2'
ns['hashlib']=types.SimpleNamespace(sha256=lambda b:Hash()if b==raw else hashlib.sha256(b))
inputs=[[0]*512 for _ in range(16)];report={};events=[]
assert ns['frozen648_reference_r8669zh'](FakePath(),inputs,{999},expected,report,lambda:events.append('save'))==expected
assert report['frozen648_original_reference']['matched_full_IDs_EOS']and events==['save']
actual=[list(v)for v in expected];actual[4][7]=-9;events=[]
try:ns['frozen648_reference_r8669zh'](FakePath(),inputs,{999},actual,report,lambda:events.append('save'));raise RuntimeError('difference accepted')
except AssertionError as e:assert e.args[0][0]=='baseline_reproducibility_FAILED_OBSERVED'
assert events==['save']and report['frozen648_original_reference']['first_differences'][0]==dict(request=4,first_different_ordinal=8,expected=11,actual=-9,expected_length=128,actual_length=128)
proof['actual_frozenreference_consumer_schema_compare_fullIDs_firstdiff_save_before_assert']=True
proof['CPU_schema_fixture_scope']='Mocked rawSHA only to exercise actual consumer; no GPU/reference numerics claim.'
for fn in ['forward','generate']:
 ts=[ast.parse((r/n).read_text())for n in ['kimi_TP16PP2_I192_protocol_r8669zc.py','kimi_TP16PP2_I192_protocol_r8669zh.py']]
 fs=[next(v for v in ast.walk(t)if isinstance(v,ast.FunctionDef)and v.name==fn)for t in ts]
 assert ast.dump(fs[0],include_attributes=False)==ast.dump(fs[1],include_attributes=False)
proof['actual_forward_generate_unchanged_flags3reference_7ordinary_decode_math_timing']=True
for n in ['launch_Kimi_TP16PP2_I192_r8669zh.py','Kimi_TP16PP2_I192_binding_r8669zh.py']:
 s=(r/n).read_text();assert 'results/Kimi-I192-smoke-static-closure-r8669zg.json'in s and 'results/Kimi-I192-compact-CPU-r8669zh.json'not in s
s=(r/'modern_Kimi_TP16PP2_I192_r8669zh.py').read_text();assert 'modern_kimi_I192_workspace_r8669zc'in s and 'modern_kimi_I192_workspace_r8669zh'not in s
inside=(r/'Kimi_TP16PP2_I192_r8669zh_inside.sh').read_text();assert '2900 python'in inside and 'COMPACT_R8669ZC=1'in inside
entry=(r/'launch_Kimi_TP16PP2_I192_r8669zh.py').read_text();assert "run='TP16PP2-I1928669zh-'"in entry and "if norm(row['State']) not in TERMINAL:owned_native_failfast_r8669zh"in entry
proof['actual_nativebinding_API_compactflag_50min2900_newnonce_and_terminal_guard']=True
proof['GPU_CAPI_executed']=False
(r/'Kimi-Model-caller-CPU-r8669zh.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(proof))
