"""Same loaded target, real ordinary baseline and target-computed speculative IDs."""
def run(g):
 import ctypes,json,numpy as np,time,hashlib,pathlib
 lib,ctx,r,control,report,save=[g[k]for k in ['lib','ctx','r','control','report','save']]
 node,tail,emitter,vocab=[g[k]for k in ['node','tail','emitter','vocab']]
 def ck(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 generation=0;serial=0;report.update(pair_cases=[],paired_phase_calls=[],QA=[],oracle_is_diagnostic_only=True,trained_spec_speed_claim=False)
 f=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert hashlib.sha256(f.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625'
 fixed=json.loads(f.read_text());prompt=fixed['natural_input_ids'][0];assert len(prompt)==512;eos=set(fixed['eos_token_ids'])
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
 qa_name='multiply';qa=chat['cases'][qa_name];qa_prompt=qa['input_ids'];qa_expected=[v['generated_ids'][0]for v in known['cases']if v['task']=='multiply'and v['grouped']];assert any(len(v)==15 and v[-1]in eos for v in qa_expected);assert set(chat['eos_token_ids'])==eos
 def mode(ordinary,label):
  nonlocal generation
  out=ctypes.c_uint64();ck(lib.modern_kimi_pair_mode_r9007(ctx,int(ordinary),ctypes.byref(out)));generation=out.value
  packets=control.collect('mode:'+label,{'generation':generation,'ordinary':ordinary});assert all(p['generation']==generation and p['ordinary']==ordinary for p in packets)
 def rows():
  out=(ctypes.c_int32*4)();ck(lib.modern_kimi_pair_rows_r9006(ctx,1,out));return list(out)
 def call(ids,past,phase):
  nonlocal serial
  ids=np.asarray(ids,np.int64);N=len(ids);n_out=N if phase in ['verify','replay','continuation']else 1
  row=np.asarray([1],np.int32);pp=np.asarray([past],np.int32);total=np.asarray([past+N],np.int32);off=np.asarray([0,N],np.int32);pos=np.arange(past,past+N,dtype=np.int64)
  logits=np.full((n_out,vocab),0x7fc0,np.uint16);out=np.full(n_out,-1,np.int64);flags=np.asarray([-1],np.int32)
  ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
  tick=time.perf_counter()
  if phase in ['ordinary','teacherforced']:ck(lib.modern_kimi_pair_ordinary_forward_r9007(ctx,generation,N,1,3,*[ptr(a)for a in [row,pp,total,off,ids,pos,logits,out,flags]]))
  elif phase=='prefill':ck(lib.modern_kimi_pair_prefill_r9006(ctx,N,ptr(row),ptr(ids),ptr(pos),ptr(logits),ptr(out),ptr(flags)))
  else:
   active,shadow,actual_past,state=rows();assert actual_past==past and state in [1,3]
   read=np.asarray([active],np.int32);write=np.asarray([shadow],np.int32)
   ck(lib.modern_kimi_pair_forward_r9006(ctx,N,1,3,*[ptr(a)for a in [read,write,row,pp,total,off,ids,pos,logits,out,flags]]))
  seconds=time.perf_counter()-tick;assert flags[0]==int(tail)+2*int(node==emitter)
  selected=None;digest=None
  if tail:
   fp=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(fp).all();argmax=np.argmax(fp,axis=-1).tolist();digest=hashlib.sha256(logits.tobytes()).hexdigest()
   if node==emitter:assert out.tolist()==argmax;selected=out.tolist()
  packets=control.collect('paircall'+str(serial),{'selected':selected,'seconds':seconds,'logits_SHA256':digest});serial+=1;selected=packets[emitter]['selected'];assert selected is not None and len(selected)==n_out
  report['paired_phase_calls'].append({'phase':phase,'N':N,'past':past,'CAPI_maxhost_seconds':max(p['seconds']for p in packets),'selected':selected,'logits_SHA256_emitter':packets[emitter]['logits_SHA256']});return selected
 def ordinary(inp,label):
  mode(True,label);tick=time.perf_counter();z=call(inp,0,'ordinary');p=len(inp);times=[time.perf_counter()-tick]
  while len(z)<128 and z[-1]not in eos:
   z+=call([z[-1]],p,'ordinary');p+=1;times.append(time.perf_counter()-tick)
   if len(z)%16==0:report['phase']=label+':'+str(len(z));save()
  assert not any(x in eos for x in z[:-1]);q={'label':label,'mode':'original_ordinary_forward','generated_ids':z,'first_EOS':next((i for i,x in enumerate(z)if x in eos),None),'timestamps_emitter_clock':times if node==emitter else None};report['pair_cases'].append(q);save();return z
 def paired(inp,gold,intended,label):
  D0=gold[1:min(len(gold),8)]
  if intended<7 and intended<len(D0):D0=D0.copy();D0[intended]=(D0[intended]+1)%vocab
  forced_inputs=[gold[0]]+D0
  mode(True,label+':same-proposed-prefix-ordinary');anchor=call(inp,0,'ordinary');assert anchor==gold[:1];teacher=[]
  for j,token in enumerate(forced_inputs):teacher+=call([token],len(inp)+j,'teacherforced')
  report.setdefault('teacherforced_same_prefix',[]).append({'label':label,'input_ids':forced_inputs,'ordinary_selected_by_position':teacher,'scope':'actual oldordinaryforward,includes positions after deliberatelywrongproposal;no greedy-continuation substitution'})
  mode(False,label);z=call(inp,0,'prefill');assert z==gold[:1];p=len(inp);first=True;rounds=[]
  while len(z)<128 and z[-1]not in eos:
   D=D0 if first else []
   inputs=[z[-1]]+D;before=rows();out=call(inputs,p,'verify'if first else'continuation');m=0
   if first:assert out==teacher,('allposition_same_proposed_prefix_greedy',label,out,teacher)
   while m<len(D)and D[m]==out[m]and D[m]not in eos:m+=1
   emitted=D[:m]+[out[m]];emitted=emitted[:128-len(z)]
   if any(v in eos for v in emitted):emitted=emitted[:next(j+1 for j,v in enumerate(emitted)if v in eos)]
   continuation=emitted[-1]not in eos and len(z)+len(emitted)<128
   ck(lib.modern_kimi_pair_commit_r9006(ctx,1,m,int(continuation)));after_commit=rows();replayed=False
   if continuation and m+1<len(inputs):
    assert after_commit[:3]==before[:3]and after_commit[3]==3
    replay=call(inputs[:m+1],p,'replay');assert replay[-1]==out[m],('replay_target_correction',label,m,replay[-1],out[m]);replayed=True
   after=rows()
   if continuation:assert after[:3]==[before[1],before[0],p+m+1]and after[3]==1
   else:assert after[3]==4
   rounds.append({'intended_prefix':intended if first else None,'actual_prefix':m,'input_width':len(inputs),'before':before,'after_commit':after_commit,'after':after,'replayed':replayed,'emitted':emitted,'continuation':continuation,'target_EOS_terminal_skip':emitted[-1]in eos,'same_proposed_prefix_ordinary_IDs':teacher if first else None})
   z+=emitted;p+=m+1;first=False
   if len(z)%16==0:report['phase']=label+':'+str(len(z));save()
  exact=z==gold;q={'label':label,'mode':'paired_target_recomputed_greedy','intended_prefix':intended,'generated_ids':z,'complete_IDs_firstEOS_equal_samecontext_AR':exact,'rounds':rounds,'first_EOS':next((i for i,v in enumerate(z)if v in eos),None)};report['pair_cases'].append(q);save();assert exact,('final_target_IDs_EOS',label,z,gold);return q
 natural=ordinary(prompt,'AR-natural512');cross=ordinary(prompt[:255],'AR-prefix255');qa_gold=ordinary(qa_prompt,'QA-before:'+qa_name)
 key='sort'if qa_name.startswith('sort_probe')else qa_name;expected=[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']];assert qa_gold in expected;report['QA'].append({'epoch':'before','task':qa_name,'complete_frozen_IDs_exact':True})
 paired(prompt,natural,7,'paired-natural512')
 for m in range(8):paired(prompt[:255],cross,m,'page255-prefix'+str(m))
 q=paired(qa_prompt,qa_gold,7,'paired-QA-EOS');assert q['first_EOS']is not None, 'actual target EOS coverage required'
 restore=ordinary(prompt,'AR-restore512');assert restore==natural
 qa_after=ordinary(qa_prompt,'QA-after:'+qa_name);assert qa_after==qa_gold;report['QA'].append({'epoch':'after','task':qa_name,'complete_frozen_IDs_exact':True})
 probes=[v for v in report['pair_cases']if v['label'].startswith('page255-prefix')];assert len(probes)==8
 assert [v['rounds'][0]['actual_prefix']for v in probes]==list(range(8)), 'forced diagnostic coverage incomplete; no output override permitted'
 report.update(serving_protocol_complete=True,trained_spec_speed_claim=False,allposition_finalIDs_EOS_pass=True,actual_page255_crossing_tested=True,actual_EOS_skip_tested=any(a['target_EOS_terminal_skip']for a in q['rounds']));save()
