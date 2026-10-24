"""Read original frozen reports and BF16 checkpoints only; no Torch/GPU."""
import array,hashlib,json,math,pathlib,struct
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='prefill-5845-f9fef807b1fc4e04870b'
p=r/'results'/f'glm-transient-Model-r5848-{run}-stage1-rank0.json';q=json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload=r/'control'/f'glm-transient-Model-r5848-control-{run}-ids.json';inputs=json.loads(payload.read_text())['I512_input_ids'];assert len(inputs)==16 and all(len(x)==512 for x in inputs)
names=['perf-I512-O8-r1-0','perf-I512-N4-r1-1'];rows={};receipts=[];source={}
lut=[struct.unpack('<f',struct.pack('<I',i<<16))[0]for i in range(65536)]
for phase in names:
 case=next((c for c in q['multi_batch_cases']if c['phase']==phase),None);source[phase]={'normal_case_saved':case is not None}
 rows[phase]={}
 for cp in q['raw_checkpoints']:
  if cp['phase']!=phase or cp['step']!=0:continue
  f=r/'results'/cp['file'];assert sha(f)==cp['sha256'];b=f.read_bytes();n,vocab=cp['shape'];assert len(b)==n*vocab*2 and n==len(cp['rows'])
  a=array.array('H');a.frombytes(b)
  receipts.append({'phase':phase,'file':cp['file'],'SHA256':sha(f),'bytes':len(b),'global_rows':cp['rows'],'shape':cp['shape']})
  for i,row in enumerate(cp['rows']):
   values=[lut[x]for x in a[i*vocab:(i+1)*vocab]];assert all(math.isfinite(x)for x in values)
   top=max(range(vocab),key=values.__getitem__);second=max((j for j in range(vocab)if j!=top),key=values.__getitem__)
   assert top==cp['GPU_argmax'][i];assert case is None or top==case['generated_ids'][row][0]
   rows[phase][row]={'v':values,'top':top,'second':second,'raw_SHA256':hashlib.sha256(b[i*vocab*2:(i+1)*vocab*2]).hexdigest()}
 assert set(rows[phase])==set(range(16))
out={'CPU_only':True,'no_new_capture':True,'same_history_scope':'Only step0 on same completeI512logicalprompt; later token histories diverge and are excluded.','report_SHA256':sha(p),'payload_SHA256':sha(payload),'raw_receipts':receipts,'phases':source,'rows':[]}
for row in range(16):
 o=rows[names[0]][row];n=rows[names[1]][row];gold=o['top'];actual=n['top'];ov=o['v'];nv=n['v'];diff=[a-b for a,b in zip(nv,ov)]
 out['rows'].append({'global_row':row,'prompt_SHA256_i64le':hashlib.sha256(struct.pack('<512q',*inputs[row])).hexdigest(),'O8_top2':[o['top'],o['second']],'O8_top2_logits':[ov[o['top']],ov[o['second']]],'N4_top2':[n['top'],n['second']],'N4_top2_logits':[nv[n['top']],nv[n['second']]],'first_token_equal':gold==actual,'N4_gold_minus_actual_logit':nv[gold]-nv[actual],'O8_gold_minus_N4actual_logit':ov[gold]-ov[actual],'same_row_max_abs':max(map(abs,diff)),'same_row_relative_L2':math.sqrt(sum(x*x for x in diff)/max(sum(x*x for x in ov),1e-300)),'raw_exact_O8_rows':[j for j,a in rows[names[0]].items()if a['raw_SHA256']==n['raw_SHA256']],'N4_top1_matching_O8_top1_rows':[j for j,a in rows[names[0]].items()if a['top']==actual],'prompt_exact_other_rows':[j for j,v in enumerate(inputs)if v==inputs[row]]})
 distances=[sum((a-b)**2 for a,b in zip(nv,rows[names[0]][j]['v']))for j in range(16)]
 out['rows'][-1]['nearest_O8_row_by_full_logits_L2']=min(range(16),key=distances.__getitem__)
 out['rows'][-1]['nearest_vs_same_squared_distance_ratio']=min(distances)/distances[row]
out['bad_first_rows']=[x['global_row']for x in out['rows']if not x['first_token_equal']]
out['prompt_hashes_unique']=len({x['prompt_SHA256_i64le']for x in out['rows']})
out['causal_native_branch']='UNKNOWN; finalstep0logits do not identify earliest wrong layer/provider/cache/reset. No token-ID magnitude argument.'
build=r/'candidates/glm-shared-validation-model-r5817/CPU-build.json';proof=json.loads(build.read_text());assert proof['library_SHA256']=='1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d'
src=r/'candidates/glm-shared-validation-model-r5817/src/glm_shared_validation_cache_r5814.cpp';assert sha(src)==proof['source_SHA256']=='4d7f4a9293cff492c1e8835560d345ea7aeace58d11b2b303188b312a0934962'
out['actual_bridge']={'build_receipt_SHA256':sha(build),'source':str(src),'source_SHA256':sha(src),'guard_lines':[{ 'line':i,'source':v}for i,v in enumerate(src.read_text().splitlines(),1)if any(s in v for s in ['geometry5775=','candidate5775=','cross5775=','tokens==1','mode5775==2'])]}
print(json.dumps(out))
