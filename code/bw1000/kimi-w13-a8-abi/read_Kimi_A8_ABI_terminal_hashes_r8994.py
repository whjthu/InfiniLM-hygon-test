import pathlib,hashlib,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-A8-ABI-real-r8994';w=json.loads((r/(label+'-watch.json')).read_text());assert w['actual_handle']=='910374.957'and w['terminal']['State']=='COMPLETED'
names=[r/'logs'/(label+'-910374.957.log'),r/'results'/(label+'-910374.957.json'),r/(label+'-watch.json'),r/(label+'-launch.json'),r/'incoming/manifest_GPU_r8994.json',r/'incoming/Kimi_A8_ABI_real_r8993.py',r/'candidates/Kimi-W13-A8-ABI-r8993/libKimi_A8_ABI_r8993.so']
print(json.dumps({'CPU_only':True,'no_new_audit_or_post':True,'actual_handle':w['actual_handle'],'files':{str(p):{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in names}}))
