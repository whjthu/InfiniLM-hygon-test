import pathlib,json,hashlib,ast
r=pathlib.Path.cwd();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=r/'Deep-latent-Model-manifest-r8943.json';m=json.loads(mp.read_text())
for key in ['CPU_consumer_contract','CPU_exception_contract']:m[key]['SHA256']=sha(r/m[key]['path'])
m['status']='CPU_READY_PENDING_ACTUALREAD_SOURCECLAIM_ROOT_GPU_GO';mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
for n,h in m['incoming_source_SHA256'].items():assert sha(r/n)==h;n.endswith('.py')and ast.parse((r/n).read_text())
print(json.dumps({'entry_SHA256':m['incoming_source_SHA256']['modern_Deep_latent_Model_r8943.py'],'manifest_SHA256':sha(mp),'mandatory3_budget_seconds':m['budget']['mandatory_three_zero_gain_seconds'],'mandatory3_margin':m['budget']['mandatory_three_zero_gain_margin_seconds']}))
