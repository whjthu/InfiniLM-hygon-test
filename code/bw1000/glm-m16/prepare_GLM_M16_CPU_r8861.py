import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bound=json.loads((r/'GLM-M16-bound-source-r8861.json').read_text())
assert bound['DSO_SHA256']=='68acfe089f757014e564c0d0ca5c896ac1f47cd98fa3a843df8b2dd5bf8c63c6'
policy=r/'GLM_PP2_B32_r8843_glm_model_resident_policy_r5640.hpp'
assert sha(policy)=='63ef7f9ceee8f51ea52b582f71d1274c9268217e4560d87d04c9e891c070246c'
s=policy.read_text();assert s.count('requests.size()>8')==2
s=s.replace('requests.size()>8','requests.size()>16').replace('cohort must be1..8','cohort must be1..16')
(r/'GLM_M16_policy_r8861.hpp').write_bytes(s.encode())
producer=r/'GLM_PP2_B32_r8843_glm_prefill_prefix_cache_r8703.cpp'
assert sha(producer)=='8dadcf96c82d7e7ac37c3bd66d721191feb21dc67d0f18cec1c54a3021ebb7ea'
s=producer.read_text();old='bool shared5640=resident5640&&requests.size()==8&&tokens==8&&';assert s.count(old)==1
s=s.replace(old,'bool shared5640=resident5640&&(requests.size()==8||requests.size()==16)&&tokens==requests.size()&&')
(r/'GLM_M16_prefix_r8861.cpp').write_bytes(s.encode())
vector=r/'glm_vector_dispatch_r8540.cpp';assert sha(vector)=='3a39b4916568d577e8fe7ee994fc1db18754fce90ea8a63d5b1b9079c047140e'
s=vector.read_text();assert s.count('(routes==32||routes==64)')==1;s=s.replace('(routes==32||routes==64)','(routes==32||routes==64||routes==128)')
(r/'GLM_M16_vector_dispatch_r8861.cpp').write_bytes(s.encode())
for a,b in [('launch_GLM_exact_budget_CPU_r8854.py','launch_GLM_M16_CPU_r8861.py'),('GLM_exact_budget_CPU_r8854_inside.sh','GLM_M16_CPU_r8861_inside.sh'),('invoke_GLM_exact_budget_CPU_r8854.py','invoke_GLM_M16_CPU_r8861.py'),('observe_GLM_exact_budget_CPU_r8854.py','observe_GLM_M16_CPU_r8861.py')]:
 s=(r/a).read_text().replace('exact_budget','M16').replace('exact-budget','M16').replace('r8854','r8861')
 (r/b).write_bytes(s.encode())
names=['GLM_M16_policy_r8861.hpp','GLM_M16_prefix_r8861.cpp','GLM_M16_vector_dispatch_r8861.cpp','build_GLM_M16_CPU_r8861.py','check_GLM_M16_r8861.cpp','launch_GLM_M16_CPU_r8861.py','GLM_M16_CPU_r8861_inside.sh']
for n in names:
 if n.endswith('.py'):ast.parse((r/n).read_text())
m={'CPU_only':True,'GPU':False,'dlopen':False,'parent_DSO_SHA256':bound['DSO_SHA256'],'sources_SHA256':{n:sha(r/n) for n in names},'bound_object_map':bound['objects'],'scope':'M16 only: Slots cohort<=16 in six actual MD consumers; resident shared exact-row producer accepts8/16; current vector dispatch routes128. M8 unchanged. Existing CUDA/SDK/math/bank allocation/admission/lease unchanged. No M32, GPU, weights, or Model performance claim.'}
(r/'GLM-M16-CPU-manifest-r8861.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
print(json.dumps({'manifest_SHA256':sha(r/'GLM-M16-CPU-manifest-r8861.json'),'files':names}))
