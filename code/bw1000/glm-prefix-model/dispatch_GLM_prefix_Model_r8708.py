import pathlib,json,hashlib,datetime,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=i/'GLM-CROSSREQUEST-Model-source-manifest-r8708.json';h=hashlib.sha256(m.read_bytes()).hexdigest();pub=json.loads((i/'GLM-prefix-Model-publication-r8708.json').read_text());assert pub['ordinary_push_success']and pub['manifest_SHA256']==h
for n,v in json.loads(m.read_text())['source_SHA256'].items():assert hashlib.sha256((i/n).read_bytes()).hexdigest()==v,n
p=r/'results/GLM-prefix-Model-r8708-entry.json';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':pub['source_commit'],'manifest_SHA256':h,'one_weight_load':True,'no_replay':True}
with p.open('x')as f:json.dump(d,f,indent=2)
with(r/'logs/GLM-prefix-Model-r8708-entry-controller.log').open('x')as f:child=subprocess.Popen(['python3',str(i/'launch_GLM_prefill_Model_r8708.py'),h],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['launcher_pid']=child.pid;p.write_text(json.dumps(d,indent=2));print(json.dumps(d))
