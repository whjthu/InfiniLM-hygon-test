"""r347 source/one-shot contract specialized to ordinary TP8PP4, no draft."""
from pathlib import Path
import hashlib,json,math,re,os
TARGET_NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];CPU_NODE='a14r2n20'
PARENT=910380;CPU_PARENT=910407
GPU_SECONDS,NATIVE_SECONDS,SHELL_SECONDS,CPU_SECONDS=5400,5100,5280,6000
PORTS=[32947,39947]
PROMPT_SHA=['66090df28f4a0b8024a03f769dd9d1f76c91a0348ea7141a31c48ae96f373934','ffcad0cc15164ae3c697434fc93772ffb86df5a7d1c4d608dfc0894dad024054']
NATIVE_SHA='6eb2bd61b5bfc0c5df3fe809bf37f065de2832be07765554db86e512cb6cd1ba'
RUNTIME_SHA='971d2b049d24019a662017a7c81a15db85c86ea1d3f81b38a92ad4535aa5c46f'
CPU825_SHA='cb3f1db0e0cd9578c77581a379d35f6683d91d6cfdb7556365a97b28ae16ed80'
CPU826_SHA='d89eee95fce8a5322781087202abec482e1576e79908349c22bc9e861d1eb9b9'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish_once(path,value):
 with Path(path).open('x',encoding='utf8',newline='\n')as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def validate_plan(p,execution=False):
 assert p['schema']==1 and re.fullmatch('[a-z0-9][a-z0-9-]{7,39}',p['run'])
 assert p['target_parent']==PARENT and p['target_nodes']==TARGET_NODES and p['CPU_parent']==CPU_PARENT and p['CPU_node']==CPU_NODE
 for key,value in dict(target_cards=32,draft_cards=0,TP=8,PP=4,C=1,context=32768,output_budget=128,requests=2,GPU_max_s=GPU_SECONDS,Native_max_s=NATIVE_SECONDS,shell_max_s=SHELL_SECONDS,CPU_max_s=CPU_SECONDS,CPUs_per_node=64,host_mem_MiB_per_node=128000).items():assert type(p[key])is int and p[key]==value,key
 assert p['PP_cuts']==[0,24,47,71,93]and p['request_order']==['retained8635','near32k']and p['prompt_tokens']==[8635,32638]
 assert p['prompt_ID_SHA256']==PROMPT_SHA
 assert p['ports']==PORTS and p['runtime_source_manifest_SHA256']==RUNTIME_SHA and p['Native_SHA256']==NATIVE_SHA
 assert p['Native_CPU_receipt_SHA256']==CPU825_SHA and p['runtime_CPU_receipt_SHA256']==CPU826_SHA
 assert p['retry']is False and p['reload']is False and p['profile']is False
 assert p['Math0']and p['MOE_EXECUTION']==p['I192_GROUPED']==0 and p['flags']==3
 assert re.fullmatch('[0-9a-f]{64}',p['controller_source_manifest_SHA256'])
 if execution:
  assert p['GPU_GO']is True and p['Model_GO']is True and p['execute_once']is True
  assert re.fullmatch('[0-9a-f]{40}',p['publication_commit'])
 return p
def verify_bundle(directory,digest):
 d=Path(directory);mf=d/'SOURCE_MANIFEST.json';assert sha(mf)==digest;m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert not Path(n).is_absolute()and'..'not in Path(n).parts and sha(d/n)==h,n
 return m
def environment(plan,nonce,epoch,head,cancel,claim):
 validate_plan(plan,False);assert len(nonce)>=32 and math.isfinite(epoch)and epoch>0
 return dict(KIMI410_ORDINARY_RUNTIME='1',KIMI410_RUN=plan['run'],KIMI410_NONCE=nonce,KIMI410_GPU_START_UNIX=str(epoch),KIMI410_NATIVE_SECONDS=str(NATIVE_SECONDS),KIMI410_NATIVE_DEADLINE_UNIX=str(epoch+NATIVE_SECONDS),KIMI410_SHELL_DEADLINE_UNIX=str(epoch+SHELL_SECONDS),KIMI410_CANCEL_FILE='/refactor/control/'+Path(cancel).name,KIMI410_RUNTIME_SOURCE_SHA256=RUNTIME_SHA,KIMI410_PUBLISHED_MODEL_CLAIM='/refactor/control/'+Path(claim).name,KIMI410_PUBLISHED_MODEL_CLAIM_SHA256=sha(claim),KIMI_ENGINE_HEAD=head,GLOBAL_PORT_R2354=str(PORTS[0]),GLOBAL_RUN_R2354=plan['run'],GLOBAL_MODEL_MODE_R2354='shca',WAVE_INSIDE_R2323='Kimi-TP8PP4-ordinary-r410/controller-v3/GPU_inside.sh')
