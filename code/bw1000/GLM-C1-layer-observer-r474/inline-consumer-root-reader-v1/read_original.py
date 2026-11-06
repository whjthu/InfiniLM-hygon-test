"""Read already-private original32 MD and five source/recipe bytes; no subprocess/Native/GPU."""
from pathlib import Path
import json,hashlib,shlex,argparse
H=lambda b:hashlib.sha256(b).hexdigest()
def read(local,repo):
 local,repo=Path(local),Path(repo)
 raw=(local/'GLM474-original32-object-MD-private.json').read_bytes();assert H(raw)=='9607c420001482f65a67593b8204144fd1dec8f228eadc52b31f61e84d0a49b8';a=json.loads(raw)
 b=(local/'GLM474-original5-source-and-r131-recipe-private.json').read_bytes();assert H(b)=='b0dfa876fc9b4c2469bf592e20ec3581997872fcb6a02dfcba32bad9aeca645a';s=json.loads(b)
 old=json.loads(json.loads((local/'GLM470-CPU1087-original-receipt-root-reader-private.json').read_bytes())['out'])['receipt']
 new=[v['receipt']for v in json.loads(json.loads((local/'Systematic-GLM471-Kimi-v2-two-CPU-first-original-terminal-observation.json').read_bytes())['out'])['originals']if v['receipt']['actual_CPU']=='910407.1101'][0]
 plan=json.loads((repo/'code/bw1000/GLM952-global-TP16-component-r470/full-native-source-v1/BUILD_PLAN.json').read_bytes())
 pins={v['object']:v['MD_SHA256']for v in plan['old_MD_records']};pins.update({v['object']:v['MD_SHA256']for v in old['compiled'].values()});pins.update({v['object']:v['MD_SHA256']for v in new['compiled'].values()})
 commands=[v['argv']for r in [old,new]for v in r['commands']];commands+=s['r131']['value']['commands']
 records=[];consumers=[]
 assert len(a['units'])==32 and len({v['object']for v in a['units']})==32
 for u in a['units']:
  assert u['live_object_match']and len(u['MD'])==1
  m=u['MD'][0];assert H(m['text'].encode())==m['SHA256']==pins[u['object']]
  dep=shlex.split(m['text'].replace('\\\n',' ').split(':',1)[1]);assert len(dep)==m['dependency_count']
  obs=[v for v in dep if v.endswith('/glm_sparse_observer_r131.hpp')];assert obs==m['observer_paths']
  r=dict(object=u['object'],object_SHA256=u['object_SHA256'],MD_path=m['path'],MD_SHA256=m['SHA256'],source=dep[0],dependencies=len(dep),observer_paths=obs,historical_MD_SHA256_matched=True)
  if obs:
   so=[v for v in s['sources']if v['path']==dep[0]];assert len(so)==1;so=so[0];assert H(so['text'].encode())==so['SHA256'];r['source_SHA256']=so['SHA256']
   cmd=[v for v in commands if isinstance(v,list)and'-c'in v and'-o'in v and v[v.index('-c')+1]==dep[0]and v[v.index('-o')+1]==u['object']];assert len(cmd)==1,(u['object'],len(cmd));r['original_compile_argv']=cmd[0]
   assert '-O2'in cmd[0]and'-D_GLIBCXX_USE_CXX11_ABI=1'in cmd[0]and cmd[0][0]=='/usr/bin/g++'
   consumers.append(r)
  records.append(r)
 assert len(consumers)==5
 return dict(schema='GLM-r474-original32-inline-consumers-v1',actual_original_MD32=True,all32_prior_MD_SHA256_matched=True,actual_original_object32_live_SHA256_matched=True,observer_consumers=consumers,records=records,read_only=True,new_compiler_calls=0,GPU=False,CPU_GO=False,GPU_GO=False,Model_GO=False,limits=['Original MD reverse graph plus five exact source/argv evidence; no new fresh-M or compile.','Need private quote-origin mirrors and all five consumers see one new header definition; no 1HOST31reuse guess.','Current fullpath event-count bound and actual512event driver memory remain unresolved; keep8GiB.','No current full nonproject header/DTK live rehash, no Native/SDK/numeric/profile qualification.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);x=p.parse_args();r=read(x.local,x.repo);Path(x.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(original32_MD=True,consumers=len(r['observer_consumers']),CPU_GO=False,GPU_GO=False)))
