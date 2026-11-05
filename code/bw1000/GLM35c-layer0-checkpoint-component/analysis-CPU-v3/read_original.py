"""Recompute only original CPU-v3 source/ELF metadata; never execute a DSO."""
from pathlib import Path
from hashlib import sha256
import json,re,argparse
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
C=R+'/candidates/GLM35c-layer0-checkpoint-component-CPU-v3'
P=R+'/incoming/GLM35c-layer0-checkpoint-component/source-v3'
MF='d09cd160b5ea25f6d343a3e820ca22ec1cceb6a730f62ea8693d7627c1f9817b'
CLAIM='52dd6f0313a1622271d7b2215b5f007c84c1a2ebeb77395775216800383d7652'
def audit(path):
 wrapper=json.loads(Path(path).read_bytes());assert wrapper['rc']==0,'transport result is not an original CPU observation'
 q=json.loads(wrapper['out']);files=q['files']
 for p,v in files.items():
  b=v['rawtext'].encode();assert len(b)==v['bytes']and sha256(b).hexdigest()==v['SHA256'],p
 def raw(p):return files[p]['rawtext']
 def obj(p):return json.loads(raw(p))
 receipt=obj(C+'/result.json');actual=receipt['actual_CPU'];rows=[r.split('|')for r in q['accounting_rows']]
 rows=[r for r in rows if r[0]==actual and r[1].strip()=='GLM35c-layer0-checkpoint-component-CPU-v3'];assert len(rows)==1
 row=rows[0];assert row[4]=='a14r2n20'
 assert receipt['source_manifest_SHA256']==MF and receipt['claim_SHA256']==CLAIM
 assert all(receipt[k]is False for k in ['GPU','Model','Native_execution','dlopen'])
 assert files[P+'/SOURCE_MANIFEST.json']['SHA256']==MF and files[P+'/CPU_RESOURCE_CLAIM.json']['SHA256']==CLAIM
 out=dict(artifact_integrity=True,actual_CPU=actual,state=row[2],exit=row[3],node=row[4],receipt_SHA256=files[C+'/result.json']['SHA256'],source_manifest_SHA256=MF,claim_SHA256=CLAIM,completed=receipt['completed'],passed=receipt['passed'],phase=receipt['phase'],GPU=False,Native_execution=False,Model_qualification=False,device_qualification=False)
 if not(receipt['completed']and receipt['passed']):
  assert row[2]in ['FAILED','TIMEOUT','CANCELLED']or row[2].startswith('CANCELLED')
  out.update(error=receipt.get('error'),qualification=False,scope='Original bounded CPU failure; partial compiled artifacts do not establish full ABI/device closure')
  return out
 assert row[2]=='COMPLETED'and row[3]=='0:0'
 assert receipt['new_HOST']==1 and receipt['retained_objects']==29 and receipt['new_HIP']==0
 assert receipt['CPU1051_remains_failed']is True and receipt['reused_CPU1051_caller_object_SHA256']=='4b9af6eda5e9a5ac891f79d6d8e6bcbec5a98896b94188f950a459059d799d25'
 assert receipt['reused_CPU1051_source_manifest_SHA256']=='c9470f1bc1d0db9e8e526a7fe6373efb622588f73a1b212a274bcf70346d5688'
 md=raw(C+'/new1.d').replace('\\\n',' ').split(':',1)[1].split();assert receipt['fresh_dependency_counts']==[len(set(md))]
 def exports(t):
  d={}
  for s in t.splitlines():
   _,kind,name=s.split()
   if kind in {'W','V','u','w','v'}and name.startswith('_Z'):continue
   assert name not in d;d[name]=kind
  return d
 before=exports(raw(C+'/parent-nm.stdout'));after=exports(raw(C+'/candidate-nm.stdout'))
 added={'glm_l0_'+n for n in ['error','id_size','id','create','keys','load','finish','step','snapshot','read','reset','destroy']}
 assert all(after.get(k)==v for k,v in before.items())and set(after)-set(before)==added
 assert all(after[k]=='T'for k in added)
 dyn=lambda t:re.findall(r'\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[([^\]]+)\]',t)
 assert dyn(raw(C+'/parent-readelf.stdout'))==dyn(raw(C+'/candidate-readelf.stdout'))
 assert not re.search(r'undefined\s+symbol|not\s+found',raw(C+'/ldd-r.stdout')+'\n'+raw(C+'/ldd-r.stderr'),re.IGNORECASE)
 assert receipt['typed_high64_recording_passed']is True and receipt['recording_Native_execution']is False
 assert 0<=receipt['inside_elapsed_seconds']<=270
 def host(p):
  assert p.startswith('/refactor/');return R+p[len('/refactor'):]
 for p,h in receipt['objects_SHA256'].items():assert q['artifacts'][host(p)]['SHA256']==h,p
 assert q['artifacts'][host(receipt['library'])]['SHA256']==receipt['library_SHA256']
 out.update(qualification=True,library=receipt['library'],library_SHA256=receipt['library_SHA256'],new_HOST=1,retained_objects=29,total_link_objects=30,original_strong_exports=len(before),new_CABI=sorted(added),fresh_compile_MD_count=len(set(md)),fresh_M_set_equality='Enforced by the pinned actually executed build consumer; only final compileMD is retained in this observation',typed_scope='libffi recording create/finish/reset high64 only; no actual12CABI/Native execution',scope='CPU source/dependency/link/ELF closure only; not genuine checkpoint/device/Model proof')
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('observation',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=(json.dumps(audit(a.observation),ensure_ascii=False,indent=2)+'\n').encode()
 with a.output.open('xb')as f:f.write(b)
 print(sha256(b).hexdigest())
