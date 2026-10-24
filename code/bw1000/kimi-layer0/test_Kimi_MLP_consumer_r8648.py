from pathlib import Path
import json,hashlib,ast,importlib.util,tempfile,sys,copy
import numpy as np
r=Path('.').resolve();p=r/'kimi_embedding_layer0_MLP_runner_r8648.py';spec=importlib.util.spec_from_file_location('mlp_contract8647',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
m=json.loads((r/'Kimi-MLP-layer0-operands-r8648.json').read_text());old=json.loads((r/'Kimi751-offline-r8641.json').read_text());checks={}
for label in ['Oa','Nca','Ncb']:
 checks[label]=all(all(mod.observer_match751(row,m['observer_reference751'][label][str(row['boundary_records'][0]['rank'])]).values())for row in old['cases'][label])
assert all(checks.values());bad=copy.deepcopy(old['cases']['Nca'][0]);next(v for v in bad['boundary_records']if v['point']=='ffn_output')['requests'][0]['fnv1a64']='injected_not_actual';assert not all(mod.observer_match751(bad,m['observer_reference751']['Nca']['0']).values())
with tempfile.TemporaryDirectory(dir=r,prefix='Kimi-MLP-CPU-contract-')as temp:
 d=Path(temp).resolve();assert d.parent==r
 records=[]
 for name,width in [('gate',4224),('up',4224),('activated',4224),('row_reduced',7168)]:
  shape=[1,512,width];filename=name+'-bf16.bin';raw=np.zeros(shape,dtype='<u2');raw.tofile(d/filename);stride=[512*8448,8448,1]if name in ['gate','up']else[512*width,width,1]
  records.append({'schema':'mlp8645-full-real','rank':0,'call':0,'point':name,'dtype':'BF16','shape':shape,'strides':stride,'packed_strides':[512*width,width,1],'bytes':raw.nbytes,'file':filename,'nonfinite':0})
 journal=d/'mlp-rank0.jsonl';journal.write_text('\n'.join(json.dumps(v)for v in records)+'\n');parsed=mod.inspect_mlp(d,0);assert set(parsed)=={'gate','up','activated','row_reduced'}
 records[0]['strides'][1]=4224;journal.write_text('\n'.join(json.dumps(v)for v in records)+'\n')
 try:mod.inspect_mlp(d,0)
 except AssertionError:checks['candidate_contiguous_layout_rejected']=True
 else:raise AssertionError('bad stride fixture accepted')
 records[0]['strides'][1]=8448;records[1]['bytes']+=2;journal.write_text('\n'.join(json.dumps(v)for v in records)+'\n')
 try:mod.inspect_mlp(d,0)
 except AssertionError:checks['short_raw_rejected']=True
 else:raise AssertionError('bad byte count accepted')
checks.update(CPU_synthetic_fixture_only=True,old751_reference_actual_metadata=True,GPU=False,Model=False,not_performance=True)
out={'passed':True,'checks':checks,'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Actual new consumer function CPU contracts; tiny zero BF16 fixture checks parser/layout/bytes only, never numerical model data or performance.'};(r/'Kimi-MLP-consumer-CPU-r8648.json').write_text(json.dumps(out,indent=2),encoding='utf8',newline='\n');print(json.dumps(out))
