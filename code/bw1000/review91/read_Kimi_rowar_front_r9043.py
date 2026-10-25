import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');tag='Kimi-layer0-ownerfront-r9043';rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910375','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if len(x.split('|'))>5 and x.split('|')[1]==tag];q={'sacct':rows,'records':[],'GPU':False,'dlopen':False}
if rows and rows[0][2]=='COMPLETED':
 for node in range(2):
  p=r/'logs'/f'{tag}-{rows[0][0]}-{node}.log';records=[json.loads(x)for x in p.read_text().splitlines()if x.startswith('{')];assert len(records)==1;d=records[0];assert d['status']=='ACTUAL_WORKER_CPU_FRONTEND_PASS'and len(d['records'])==8 and all(z['inherited_mask_count']>=32 and z['width']==4 and z['ownerCount']==8 and not z['manual_affinity_mutation']for z in d['records']);q['records'].append(d)
 q['passed']=rows[0][3]=='0:0'and len(q['records'])==2;p=r/'results/Kimi-layer0-ownerfront-r9043.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));q['proof_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(q))
