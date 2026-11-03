"""Independent stdlib NPZ/FP32/BF16 reader, original GPU1371 only; no execution."""
from pathlib import Path
import argparse,ast,base64,hashlib,io,json,math,statistics,struct,zipfile
SHA=lambda b:hashlib.sha256(b).hexdigest()
ARCHIVE='74b31b059ad531338b9774f826a041ce1fe1784bd6bfa35f4df84bfa37cf27f5'
REPORT='4c542c6c1cdcd837ba2e189fea1ea2cf504dce28cbd463b4b92dc81830f43aeb'
NATIVE='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
BRIDGE='d1974a44a3bc6276725ab55d8669fd797735e7160a06fe2c5bb1c89aa611933a'
MODES=[0,1,0,1,1,0,1,0,0,1];PAIRS=[(2,3),(5,4),(7,6),(8,9)]
def F(x):
 try:return struct.unpack('<f',struct.pack('<f',x))[0]
 except OverflowError:return math.copysign(math.inf,x)
def norm(a,b):return math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))/(math.sqrt(math.fsum(y*y for y in b))+1e-30)
def bf16(x):
 u=struct.unpack('<I',struct.pack('<f',F(x)))[0]
 if(u&0x7fffffff)>0x7f800000:return(u>>16)|0x40
 return((u+0x7fff+((u>>16)&1))>>16)&0xffff
def expand(x):return struct.unpack('<f',struct.pack('<I',x<<16))[0]
def key(p):return p.split('/results/',1)[1]
def arrays(raw):
 out={}
 with zipfile.ZipFile(io.BytesIO(raw))as z:
  for name in z.namelist():
   assert name.endswith('.npy')and'/'not in name;b=z.read(name);assert b[:6]==b'\x93NUMPY'
   if b[6:8]==b'\x01\x00':n=struct.unpack('<H',b[8:10])[0];start=10
   else:assert b[6:8]in[b'\x02\x00',b'\x03\x00'];n=struct.unpack('<I',b[8:12])[0];start=12
   h=ast.literal_eval(b[start:start+n].decode().strip());payload=b[start+n:];assert h['descr']in['|u1','<u1']and h['fortran_order']is False and h['shape']==(len(payload),)
   out[name[:-4]]=payload
 return out
def decode(raw,meta,kind,shape):
 assert meta['shape']==shape and meta['dtype']=={'I8':0,'BF16':9,'F32':10,'I64':3}[kind]
 assert SHA(raw)==meta['SHA256']and len(raw)==math.prod(shape)*{'I8':1,'BF16':2,'F32':4,'I64':8}[kind]
 values=[x for(x,)in struct.iter_unpack({'I8':'<b','BF16':'<H','F32':'<f','I64':'<q'}[kind],raw)]
 if kind=='BF16':values=list(map(expand,values))
 assert all(math.isfinite(x)for x in values);return values
def activation(gu,rows):
 result=[]
 for row in range(rows):
  for col in range(256):
   g=min(gu[row*512+col],10.);u=max(-10.,min(gu[row*512+col+256],10.))
   try:ex=F(math.exp(F(-g)))
   except OverflowError:ex=math.inf
   sig=F(1/F(F(1)+ex));result.append(expand(bf16(F(F(g*sig)*u))))
 return result
def activation_check(actual,wanted):
 differences=[abs(bf16(a)-bf16(b))for a,b in zip(actual,wanted)];m=max(differences,default=0);nr=norm(actual,wanted)
 assert m<=1 and nr<=.005,(m,nr)
 return dict(NRMSE=nr,max_BF16_ULP=m,different_values=sum(x!=0 for x in differences))
def quant_check(a,p,s,rows):
 max_scale=0.;neq=0
 for row in range(rows):
  values=a[row*256:(row+1)*256];expected=F(max(max(abs(x)for x in values)/127.,F(1e-8)))
  assert s[row]>0;relative=abs(s[row]-expected)/expected;assert relative<=2e-6;max_scale=max(max_scale,relative)
  inverse=F(1/s[row])
  for col,x in enumerate(values):
   v=F(x*inverse);mag=math.floor(F(abs(v)+F(.5)));wanted=max(-127,min(127,mag if v>=0 else-mag));actual=p[row*256+col]
   if actual!=wanted:
    distance=abs(abs(v)-(math.floor(abs(v))+.5));assert abs(actual-wanted)<=1 and distance<=4*(2**-23)*max(abs(v),1);neq+=1
 return dict(max_relative_scale_error=max_scale,packed_near_half_one_bin_count=neq,half_away=True)
def metric(values):
 assert len(values)==10 and all(math.isfinite(v)and v>0 for v in values)
 off=statistics.mean(values[i]for i in [2,5,7,8]);on=statistics.mean(values[i]for i in [3,4,6,9]);ratios=[values[n]/values[o]for o,n in PAIRS]
 return dict(same_query_max8_s=values,formal_OFF_mean_s=off,formal_ON_mean_s=on,ON_over_OFF=on/off,improvement_percent=100*(1-on/off),adjacent_pairs=[dict(OFF_index=o,ON_index=n,ON_over_OFF=values[n]/values[o])for o,n in PAIRS],pairs_faster=sum(x<1 for x in ratios),block_ON_over_OFF=[statistics.mean(values[i]for i in[3,4])/statistics.mean(values[i]for i in[2,5]),statistics.mean(values[i]for i in[6,9])/statistics.mean(values[i]for i in[7,8])],OFF_endpoint_drift_signed=[values[5]/values[2]-1,values[8]/values[7]-1],cross_block_OFF_change=statistics.mean(values[i]for i in[7,8])/statistics.mean(values[i]for i in[2,5])-1)
def analyze(path):
 raw=Path(path).read_bytes();outer=json.loads(raw)
 if'sealed_archive'in outer:raw=outer['sealed_archive']['text'].encode();assert len(raw)==outer['sealed_archive']['bytes']and SHA(raw)==outer['sealed_archive']['SHA256']
 assert SHA(raw)==ARCHIVE;a=json.loads(raw);assert a['schema']=='Deep438-terminal-proof-archive-v1';files={}
 for item in a['files']:
  b=base64.b64decode(item['base64'])if'base64'in item else item['text'].encode();assert len(b)==item['bytes']and SHA(b)==item['SHA256'];k=key(item['path']);assert k not in files;files[k]=b
 assert len(files)==228 and len(a['accounting'])==4 and all('|COMPLETED|0:0|'in line for line in a['accounting'])
 rb=files['Deep438-MoE-910374.1371.json'];assert SHA(rb)==REPORT;r=json.loads(rb);w=json.loads(files['Deep-T1-MoE-activation-quant-r438-v1-watch.json'])
 assert w['completed']and w['passed']and w['normal_all8']and w['actual_GPU']=='910374.1371'and w['CPU_keeper']=='910407.892'and w['srun_exit_code']==0 and w['report_SHA256']==REPORT
 for phase in ['fresh_before','post']:
  x=w[phase];assert x['node']=='a02r4n04'and x['KFD_available']and not x['kfd_pids']and not x['matching_component_processes']and not x['port43882_listening']
 assert w['fresh_before']['actual_step']=='910374.1370'and w['post']['actual_step']=='910374.1372'
 assert r['completed']and r['passed']and r['normal']and r['Native_SHA256']==NATIVE and r['bridge_SHA256']==BRIDGE and not r['Model_loaded']and not r['phase_observer']
 assert r['actual_SDK_metadata_executed']and r['metadata_startup_returncode']==0 and not r['SDK_metadata_used_as_math_gate']and r['metadata_exe_static_GPU_queries_not_zero']
 metadata=files[key(r['metadata_stdout_path'])];assert SHA(metadata)==r['metadata_stdout_SHA256']and json.loads(metadata.decode().splitlines()[-1])==r['SDK_metadata'];assert len(r['SDK_metadata']['SDK_default_metadata_only'])==9
 assert len(r['hardware'])==8 and len({x['PCI']for x in r['hardware']})==8
 expected={(rank,'query'+str(i))for rank in range(8)for i in range(10)}|{(rank,tag)for rank in range(8)for tag in ['same-GU-original-SDK-before-candidate','same-GU-fused-after','T2-fallback-mode0','T2-fallback-mode1']}
 assert len(r['raw_archives'])==112;packets={};raw_packets={}
 for item in r['raw_archives']:
  pair=(item['rank'],item['tag']);assert pair in expected and pair not in packets;z=files[key(item['npz'])];j=files[key(item['metadata'])];assert SHA(z)==item['npz_SHA256']and SHA(j)==item['metadata_SHA256'];side=json.loads(j);npz=arrays(z)
  assert set(npz)=={k for k,v in side['fields'].items()if v is not None};decoded={};tag=item['tag'];tokens=2 if tag.startswith('T2-')else 1
  spec={0:('BF16',[tokens,4096]),1:('F32',[tokens,6]),2:('I64',[tokens,6])}
  if tag.startswith('same-GU'):
   for role,rows in enumerate([6,1]):
    first=3+role*7;spec.update({first:('BF16',[rows,512]),first+1:('BF16',[rows,256]),first+2:('I8',[rows,256]),first+3:('F32',[rows,1])})
    if tag=='same-GU-fused-after':spec.update({first+4:('BF16',[rows,256]),first+5:('I8',[rows,256]),first+6:('F32',[rows,1])})
  assert set(npz)=={str(k)for k in spec}
  for k,(kind,shape)in spec.items():decoded[k]=decode(npz[str(k)],side['fields'][str(k)],kind,shape)
  for offset in range(0,len(decoded[2]),6):assert len(set(decoded[2][offset:offset+6]))==6 and all(0<=x<256 for x in decoded[2][offset:offset+6])
  assert all(x>=0 for x in decoded[1]);packets[pair]=decoded;raw_packets[pair]=npz
 assert set(packets)==expected
 same=[];fallback=[];output_error=0.;route_variation=0;cross_owner_error=0.;affinity=[];thread_hashes=[]
 for rank,owner in enumerate(r['owners']):
  assert owner['rank']==rank and owner['loaded_key_count']==1544 and owner['normal_close']and owner['destroy_returned']and owner['Runtime_expired']and owner['same_GU_qualified']
  assert owner['actual_affinity']==owner['expected_affinity']and len(owner['actual_affinity'])==4 and len(owner['initial_affinity'])>=32;affinity+=owner['actual_affinity'];assert len(owner['calls'])==10
  before=packets[(rank,'same-GU-original-SDK-before-candidate')];after=packets[(rank,'same-GU-fused-after')]
  for k in before:assert raw_packets[(rank,'same-GU-original-SDK-before-candidate')][str(k)]==raw_packets[(rank,'same-GU-fused-after')][str(k)],('Same source lease changed',rank,k)
  for role,rows in enumerate([6,1]):
   first=3+role*7;wanted=activation(before[first],rows)
   old_check=activation_check(before[first+1],wanted);new_check=activation_check(after[first+4],wanted)
   old_quant=quant_check(before[first+1],before[first+2],before[first+3],rows);new_quant=quant_check(after[first+4],after[first+5],after[first+6],rows)
   direct=dict(activation_bits_equal=raw_packets[(rank,'same-GU-fused-after')][str(first+1)]==raw_packets[(rank,'same-GU-fused-after')][str(first+4)],packed_bits_equal=raw_packets[(rank,'same-GU-fused-after')][str(first+2)]==raw_packets[(rank,'same-GU-fused-after')][str(first+5)],scale_bits_equal=raw_packets[(rank,'same-GU-fused-after')][str(first+3)]==raw_packets[(rank,'same-GU-fused-after')][str(first+6)])
   same.append(dict(rank=rank,role=role,rows=rows,original_vs_independent_F32_BF16=old_check,fused_vs_independent_F32_BF16=new_check,original_quant=old_quant,fused_quant=new_quant,**direct))
  t0=packets[(rank,'T2-fallback-mode0')];t1=packets[(rank,'T2-fallback-mode1')];e=norm(t1[0],t0[0]);assert e<.01 and t0[2]==t1[2];fallback.append(e)
  first=packets[(rank,'query0')]
  for index,(mode,call)in enumerate(zip(MODES,owner['calls'])):
   assert call['index']==index and call['mode']==mode and call['phase']==('warm'if index<2 else'formal');s=call['stats'];assert s['routed']==s['shared']==[1,mode,1-mode]and(s['calls'],s['hits'],s['fallbacks'])==(1,mode,1-mode)
   assert math.isfinite(s['MoE_complete_s'])and s['MoE_complete_s']>=s['MoE_returned_s']>0
   assert call['forward_CAPI_s']>=s['MoE_complete_s']and call['CAPI_with_proof_s']>=call['forward_CAPI_s']and call['proof_D2H_bytes']==8264 and call['proof_D2H_sync_calls']==3
   p=packets[(rank,'query'+str(index))];error=norm(p[0],first[0]);assert error<.01;output_error=max(output_error,error);route_variation+=int(p[2]!=first[2])
   assert math.isclose(error,call['output_vs_first_OFF_NRMSE'],rel_tol=1e-10,abs_tol=1e-12)
   other=packets[(0,'query'+str(index))];assert p[2]==other[2];cross=norm(p[0],other[0]);assert cross<.002;cross_owner_error=max(cross_owner_error,cross)
  hashes={x['stats']['owner_thread_hash']for x in owner['calls']};assert len(hashes)==1;thread_hashes+=list(hashes)
 assert len(set(affinity))==32 and len(set(thread_hashes))==8
 timing={}
 for name in ['CPP_complete','forward_CAPI','CAPI_with_proof','checked_batch']:
  values=[]
  for index in range(10):
   rows=[o['calls'][index]for o in r['owners']]
   if name=='CPP_complete':v=max(x['stats']['MoE_complete_s']for x in rows)
   elif name=='checked_batch':v=max(x['measurement_end_s']for x in rows)-min(x['measurement_begin_s']for x in rows)
   else:v=max(x[name+'_s']for x in rows)
   field={'CPP_complete':'max8_CPP_complete_s','forward_CAPI':'max8_forward_CAPI_s','CAPI_with_proof':'max8_CAPI_with_proof_s','checked_batch':'same_trial_checked_batch_wall_s'}[name]
   assert math.isclose(v,r['matched_queries'][index][field],rel_tol=1e-12,abs_tol=1e-12);values.append(v)
  timing[name]=metric(values)
 journal=[json.loads(line)for line in files['Deep438-MoE-910374.1371/scalar.jsonl'].splitlines()];assert len(journal)==80 and len({(x['rank'],x['index'])for x in journal})==80
 assert r['CPU_source_weights_and_hidden_canary_unchanged']and len(r['checkpoint']['selected_weight_payload_SHA256'])==1544
 return dict(passed=True,scope='Independent raw same-GU activation/quant, complete synthetic-input learned MoE numerical comparison, same-query timing/lifecycle; not Model',archive_SHA256=ARCHIVE,report_SHA256=REPORT,raw_files=228,NPZ_sidecars=112,actual_GPU='910374.1371',CPU_keeper='910407.892',post='910374.1372',accounting=a['accounting'],same_GU_role_checks=same,all_same_GU_activation_packed_scale_bits_equal=all(x['activation_bits_equal']and x['packed_bits_equal']and x['scale_bits_equal']for x in same),same_GU_original_inputs_and_states_unchanged=True,T2_max_NRMSE=max(fallback),formal_and_warm_complete_output_max_NRMSE=output_error,query_route_ID_variations=route_variation,cross_owner_output_max_NRMSE=cross_owner_error,normal_owners=8,actual_owner_masks_disjoint=True,timing=timing,driver_wall_s=r['driver_wall_s'],SDK_device_query_startup_s=r['SDK_device_query_startup_s'],checkpoint_payload_read_hash_s=r['weight_payload_read_and_hash_s'],GPU_allocated_elapsed_s=95,GPU_cards=8,GPU_card_seconds=760,CPU_keeper_elapsed_s=102,Model_qualified=False,Model_gain='UNKNOWN',limits=['One layer16/synthetic immutable activation, actual full learned-MoE TP8 only; no hash3 device/Model/real-activation qualification','No separately recomputed full-MoE or Model oracle: independent oracle covers changed activation+quant chain','CAPI proof adds three D2H/drains; checked batch additionally archives/oracle/barriers, not production Native wall','CPU source weight/hidden canary reported with verified actual producer guards; private GPU weights are not independently read back','No scalar hardware-reference or static 258-submit count extrapolation'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');args=p.parse_args();result=analyze(args.archive);text=json.dumps(result,indent=2)+'\n'
 if args.output:Path(args.output).write_text(text,encoding='utf8')
 else:print(text)
