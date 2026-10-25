import pathlib,json,hashlib,ast
w=pathlib.Path(__file__).parent;p=w/'Deep-Cast-Model-private-r9060';p.mkdir();old=w/'Deep-sink-Model-private-r9045';cpuobs=json.loads((w/'Deep-scalar-Cast-CPU-observed-b-r9054.json').read_text());cpu=cpuobs['receipt'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();names=['modern_Deep_Cast_Model_r9060.py','Deep_Cast_Model_protocol_r9060.py','verify_Deep_Cast_Model_r9060.py','launch_Deep_Cast_Model_r9060.py','Deep_Cast_Model_r9060_inside.sh']
main=(old/'modern_Deep_sink_Model_r9045.py').read_text().replace('deep-sink-model-r9045','deep-scalar-Cast-model-r9060').replace('Deep-sink-Model-manifest-r9045.json','Deep-Cast-Model-manifest-r9060.json').replace('Deep_sink_Model_protocol_r9045','Deep_Cast_Model_protocol_r9060')
imports='import torch\nfrom safetensors import safe_open\nfrom transformers import PreTrainedTokenizerFast\n';assert main.count(imports)==1;main=main.replace(imports,'')
anchor='# Sealed build receipt retains compile-input provenance; reused DSO requires no current header equality.';assert main.count(anchor)==1
preflight='''
for group,base in [('incoming_source_SHA256',r/'incoming'),('existing_incoming_dependency_SHA256',r/'incoming'),('existing_remote_SHA256',r)]:
 for n,h in manifest[group].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
assert len(prebuild['link']['full']['objects'])==48
if os.environ.get('INFINILM_DEEP_FRONTEND_PREFLIGHT_ONLY_R9060')=='1':
 import sys
 assert 'torch'not in sys.modules
 print(json.dumps(dict(passed=True,CPU_only=True,preflight='actualfinalfrontend beforetorch/CDLL/nativeAPI',entry=__file__,argv=sys.argv,manifest_SHA256=hashlib.sha256((r/'incoming/Deep-Cast-Model-manifest-r9060.json').read_bytes()).hexdigest(),DSO_SHA256=prebuild['libraries']['full']['SHA256'],linked_objects=48,actual_fields_verified=True,no_torch=True,no_CDLL=True,no_GPU=True)),flush=True)
 raise SystemExit(0)
'''
main=main.replace(anchor,anchor+'\n'+preflight+imports)
main=main.replace("for symbol in ['deep_sink_broadcast_r9040'","for symbol in ['deep_scalar_cast_mode_r9054','deep_scalar_cast_counts_r9054','deep_sink_broadcast_r9040'")
(p/names[0]).write_text(main)
s=(old/'Deep_sink_Model_protocol_r9045.py').read_text().replace('owner_trials_r9045','owner_trials_r9060').replace('source_scope_r9045','source_scope_r9060')
s=s.replace("check(lib.deep_sink_broadcast_mode_r9040(arm))", "check(lib.deep_sink_broadcast_mode_r9040(1));check(lib.deep_scalar_cast_mode_r9054(arm))")
s=s.replace(' def select(arm):', ''' lib.deep_scalar_cast_mode_r9054.argtypes=[ctypes.c_int];lib.deep_scalar_cast_mode_r9054.restype=ctypes.c_int
 lib.deep_scalar_cast_counts_r9054.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_scalar_cast_counts_r9054.restype=ctypes.c_int
 assert lib.deep_scalar_cast_mode_r9054(2)==1
 def castcounts():
  ans=[]
  for d in range(8):
   q=(ctypes.c_uint64*3)();check(lib.deep_scalar_cast_counts_r9054(d,q,3));ans.append(list(q))
  return ans
 def select(arm):''')
s=s.replace('sink_before=sinkcounts();fused_before=', 'sink_before=sinkcounts();cast_before=castcounts();fused_before=')
s=s.replace("entry['sink_arm']=arm;entry['sink_prefill_chunk_checks']=[]", "entry['sink_arm']=1;entry['cast_arm']=arm;entry['cast_prefill_chunk_checks']=[];entry['sink_prefill_chunk_checks']=[]")
s=s.replace('sb=sinkcounts()if step==0 else None;', 'cb=castcounts()if step==0 else None;sb=sinkcounts()if step==0 else None;')
s=s.replace("se=[[v[3],8*v[4],v[4]-v[3]]if arm else[0,0,0]for v in pd]", "se=[[v[3],8*v[4],v[4]-v[3]]for v in pd]")
s=s.replace("     for v in pd:\n", """     cd=[[y-x for x,y in zip(u,v)]for u,v in zip(cb,castcounts())]
     assert cd==([[172,21844,21672]]*8 if arm else[[0,0,0]]*8),('actualCast_prefix_branch',arm,cd)
     entry['cast_prefill_chunk_checks'].append(dict(chunk=a,counter_delta=cd));save()
     for v in pd:
""")
s=s.replace("v[5]==4*v[2]+2*v[4]+2*v[3]", "v[5]==4*v[2]+2*v[4]+2*v[3]-(21672 if arm else 0)")
anchor="  entry['accepted_decode_fused_counter_delta']="
s=s.replace(anchor,"""  entry['cast_counter_delta']=[[y-x for x,y in zip(u,v)]for u,v in zip(cast_before,castcounts())]
  ce=[[sum(c['counter_delta'][d][k]for c in entry['cast_prefill_chunk_checks'])for k in range(3)]for d in range(8)]
  assert entry['cast_counter_delta']==ce==([[1376,174752,173376]]*8 if arm else[[0,0,0]]*8),('Castprefillonly/nodecodehits',entry['cast_counter_delta'],ce)
"""+anchor)
s=s.replace("report['QA_before_sink_arm']=1", "report['QA_before_sink_arm']=1;report['QA_before_cast_arm']=1")
s=s.replace('FIXED accepted967SoftmaxCastmode2 inbotharms; onlyr9040runSinkbroadcast OFF/ON, noGEMM/SDK/math change; no newdecodekernelgain expected','FIXED accepted1000mode2+sinkON inallarms; onlyr9054scalarprefixCast0/1/1/0, noGEMM/SDK/math change; no newdecodekernelgain expected')
(p/names[1]).write_text(s)
audit=(old/'verify_Deep_sink_Model_r9045.py').read_text().replace('Deep-sink-Model-manifest-r9045.json','Deep-Cast-Model-manifest-r9060.json').replace('owner_trials_r9045','owner_trials_r9060')
audit=audit.replace("assert [c['sink_arm']for c in trials]==[0,1,1,0]and q['QA_before_sink_arm']==1", "assert [c['sink_arm']for c in trials]==[1,1,1,1]and [c['cast_arm']for c in trials]==[0,1,1,0]and q['QA_before_sink_arm']==q['QA_before_cast_arm']==1")
audit=audit.replace("v[5]==4*v[2]+2*v[4]+2*v[3]", "v[5]==4*v[2]+2*v[4]+2*v[3]-(21672 if c['cast_arm']else 0)")
audit+='''
for c in trials:
 assert len(c['cast_prefill_chunk_checks'])==8
 for v in c['cast_prefill_chunk_checks']:assert v['counter_delta']==([[172,21844,21672]]*8 if c['cast_arm']else[[0,0,0]]*8)
 assert c['cast_counter_delta']==([[1376,174752,173376]]*8 if c['cast_arm']else[[0,0,0]]*8)
'''
(p/names[2]).write_text(audit)
launch=(old/'launch_Deep_sink_Model_r9045.py').read_text().replace('deep-sink-model-r9045','deep-scalar-Cast-model-r9060').replace('Deep-sink-Model-manifest-r9045.json','Deep-Cast-Model-manifest-r9060.json').replace('verify_Deep_sink_Model_r9045.py','verify_Deep_Cast_Model_r9060.py').replace('Deep_sink_Model_r9045_inside.sh','Deep_Cast_Model_r9060_inside.sh');(p/names[3]).write_text(launch)
sh=(old/'Deep_sink_Model_r9045_inside.sh').read_text().replace('modern_Deep_sink_Model_r9045.py','modern_Deep_Cast_Model_r9060.py');(p/names[4]).write_bytes(sh.replace('\r\n','\n').encode())
cpush=sh[:sh.index('python -B /refactor/incoming/deep_affinity_env_contract_r7166.py')]+"export INFINILM_DEEP_FRONTEND_PREFLIGHT_ONLY_R9060=1\ntimeout --kill-after=5s 120 python -B -u /refactor/incoming/modern_Deep_Cast_Model_r9060.py\n";(p/'CPU_Deep_Cast_frontend_r9060_inside.sh').write_bytes(cpush.replace('\r\n','\n').encode());names+=['CPU_Deep_Cast_frontend_r9060_inside.sh']
cl=(w/'Deep-scalar-Cast-private-r9054/launch_Deep_scalar_Cast_CPU_r9054.py').read_text().replace('Deep-scalar-Cast-CPU-r9054','Deep-Cast-frontend-CPU-r9060').replace('build_Deep_scalar_Cast_r9054_inside.sh','CPU_Deep_Cast_frontend_r9060_inside.sh');(p/'launch_Deep_Cast_frontend_CPU_r9060.py').write_text(cl);names+=['launch_Deep_Cast_frontend_CPU_r9060.py']
om=json.loads((old/'Deep-sink-Model-manifest-r9045.json').read_text());m=dict(hardware='BW1000',level='MODEL_PENDING',status='ROOT_ASTRA86_ONE_MODEL_EXPERIMENT_EXCEPTION_NOT_ADOPTED',cpu_receipt_path='results/deep-scalar-Cast-CPU-r9054.json',DSO_SHA256=cpu['libraries']['full']['SHA256'],incoming_source_SHA256={n:sha(p/n)for n in names},existing_incoming_dependency_SHA256=om['existing_incoming_dependency_SHA256'],existing_remote_SHA256={'results/deep-scalar-Cast-CPU-r9054.json':cpuobs['receipt_SHA256'],'candidates/deep-scalar-Cast-r9054/libdeep_scalar_Cast_r9054.so':cpu['libraries']['full']['SHA256'],'incoming/Deep-B32-samegeom-reference-r7269.json':'ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797'},resource=dict(parent='910374',node='a02r4n05',DCU=8,CPU=64,memMiB=128000,hard_seconds=2400,inner_seconds=2300),plan='Onceweightload,TP8PP1B32I512O128prefillM4decodeM32;mode2+sinkONfixed,scalarCast0/1/1/0;4full128+12QA;normalmarkerlimitationpreserved.',actual_final_argv=['timeout','--kill-after=20s','2300','python','-B','-u','/refactor/incoming/modern_Deep_Cast_Model_r9060.py'],quality='FinalIDs/firstEOS/QA/lifecycle; nointermediatebitexact gate; actualCastandfixedsink/packedOpscounters; noDecodegainpromise.',promotion='1012 remainsPARK/defaultOFF; thisexperiment requiresnew actualModelproof androotadoptiondecision.')
(p/'Deep-Cast-Model-manifest-r9060.json').write_text(json.dumps(m,indent=2))
for f in p.glob('*.py'):ast.parse(f.read_text())
for f in p.glob('*.sh'):assert b'\r'not in f.read_bytes()
print(json.dumps(dict(source_ready=True,files=names,DSO_SHA256=m['DSO_SHA256'])))
