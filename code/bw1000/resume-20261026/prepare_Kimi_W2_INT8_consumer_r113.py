import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent;d=r/'Kimi-W2-INT8-private-r113';s=(r/'Kimi-W2-pair-private-r9280/run_GPU_r9280.py').read_text();s=s.replace("sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))","sys.path.insert(0,'/refactor/incoming/Kimi-W2-pair-r9280')")
s=s.replace('modern_kimi_W2_pair_run_r9280','modern_kimi_W2_INT8_run_r113').replace('modern_kimi_W2_pair_result_r9280','modern_kimi_W2_INT8_result_r113');s=s.replace("'modern_kimi_W2_pair_counts_r9280','modern_kimi_grouped_I192_r8669y'","'modern_kimi_i8_counts_r113','modern_kimi_i8_bind_r113','modern_kimi_grouped_I192_r8669y'")
s=s.replace("len(owner['arms'])==16 and len(owner['current_branch'])==2","len(owner['arms'])==24 and len(owner['current_branch'])==3 and owner['INT8_atom_CPU_integer_oracle_all256_pass']")
s=s.replace("q['W2_original_OFF_calls']==34 and q['W2_pair_ON_calls']==34","q['W2_original_OFF_calls']==0 and q['W2_pair_ON_calls']==34 and q['I8_OFF_calls']==34 and q['I8_ON_calls']==34")
s=s.replace('KIMI_W2_PAIR_R9280_NORMAL','KIMI_W2_INT8_R113_NORMAL').replace("'/refactor/results/Kimi-W2-pair-r9280-'","'/refactor/results/Kimi-W2-INT8-r113-'")
s=s.replace("one adjacent-pair W2 candidate; same allotherb4objects/flags","I192 W2 A8group32 MMAC versus current W2pair baseline; allcosts including jobs/quant/partial/reduce; T64thirdscenario constructed896expert lowreuse; same otherb4objects/flags")
assert 'len(owner[\'arms\'])==24' in s;compile(s,'run_GPU_r113.py','exec');(d/'run_GPU_r113.py').write_text(s)
# Logical memory/layout validation, no device calls.
checks=0
for T in [8,64]:
 R=T*16;spans=[(0,R*192),(R*192,R*216),(R*216,R*(216+3584*4)),(R*(216+3584*4),R*(220+3584*4)),(R*(220+3584*4),R*(220+3584*4)+4)]
 for a,b in spans:assert a<b and a%4==0;checks+=1
 assert all(spans[i][1]==spans[i+1][0]for i in range(4));assert spans[-1][1]==R*(192+24+3584*4+4)+4
 assert len({lane+64*i for lane in range(64)for i in range(4)})==256
 for p in range(512):n,k=divmod(p,32);assert p==n*32+k and 0<=n<16 and 0<=k<32;checks+=1
ids=[x%896 for x in range(1024)];assert all(len(set(ids[t*16:t*16+16]))==16 for t in range(64));hist=[ids.count(e)for e in range(896)];assert sum(hist)==1024 and max(hist)==2 and sum(x>0 for x in hist)==896
q={'CPU_only':True,'GPU':False,'logical_layout_checks':checks,'spans_and_aliasing_checked':True,'B_colmajor':'B[k+n*32] from original packed W2[e,n,g*16+k/2], correct low/highnibble','outputtmp':'du_store rowmajor[m*16+n]; lane+64*i covers256once','constructed_T64':{'R':1024,'experts':896,'maxrows':2,'jobs16':896,'physicalrows':14336,'utilization':1024/14336,'per_token_top16unique':True},'scratch_bytes_including_canaries':{str(T):T*16*(192+24+3584*4+4)+4+512 for T in [8,64]},'run_GPU_SHA256':hashlib.sha256((d/'run_GPU_r113.py').read_bytes()).hexdigest(),'producer_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'limits':'logicalCPU check; newGPUkernel notexecuted; mandatoryatomCPUintegeroracle within futurecomponentwarm'}
(d/'CPU-logical-layout-r113.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
