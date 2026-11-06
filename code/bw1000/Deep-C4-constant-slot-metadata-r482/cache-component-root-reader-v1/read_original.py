"""Independent stdlib replay of original cache outputs/pins/counters/clocks; no Native/GPU."""
from pathlib import Path
import argparse,json,hashlib,struct,statistics,math,functools
H=lambda b:hashlib.sha256(b).hexdigest()
@functools.lru_cache(None)
def data(layer,epoch):
 # Separate periodic byte construction from production NumPy vector expressions.
 rows=[]
 for slot in range(120):
  rows.append(bytes((((17*slot+13*c+7*layer+29*epoch)%120)|(((slot+c+layer+epoch)&1)<<7))for c in range(128)))
 cycle=b''.join(rows);values=(cycle*69)[:8192*128]
 scales=(b''.join(struct.pack('<f',1+((slot+layer+3*epoch)%31)/32)for slot in range(31))*265)[:8192*4]
 assert all(v not in (127,255)for v in values)
 return values,scales
@functools.lru_cache(None)
def raw_cache(layer,epoch,count):
 v,s=data(layer,epoch);out=bytearray(b'\xa5'*(128*8448))
 for slot in range(count):
  page,row=divmod(slot,64);base=page*8448
  out[base+row*128:base+(row+1)*128]=v[slot*128:(slot+1)*128]
  out[base+8192+row*4:base+8192+(row+1)*4]=s[slot*4:(slot+1)*4]
 return bytes(out)
def indices(name):
 if name=='prefix1':return list(range(1))
 if name in ('prefix2124','reset_reseed_prefix2124'):return list(range(2124))
 if name=='prefix8192':return list(range(8192))
 if name=='reverse8192':return list(reversed(range(8192)))
 if name=='page_edges_duplicates':return[0,63,64,127,128,4095,4096,8191,0,8191]
 raise AssertionError(name)
@functools.lru_cache(None)
def gathered(layer,name):
 idx=indices(name);v,s=data(layer,int(name=='reset_reseed_prefix2124'))
 return b''.join(v[i*128:(i+1)*128]for i in idx),b''.join(s[i*4:(i+1)*4]for i in idx)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def read(local,repo):
 local,repo=Path(local),Path(repo);private=local/'Deep1534-original60-private'
 ix=json.loads((private/'original60-index.json').read_bytes());assert len(ix['files'])==60
 for r in ix['files']:
  b=(private/r['name']).read_bytes();assert len(b)==r['bytes']and H(b)==r['SHA256']
 wrapper=local/'Deep482-cache-component-v1-original-bounded-terminal-watch.json';original=json.loads(json.loads(wrapper.read_bytes())['out']);owner=original['owner']['value'];watch=original['watch']
 rows=original['accounting'];states={r[0]:r[2:5]for r in rows};assert states['910374.1534']==['COMPLETED','0:0','252']and states['910407.1118']==['COMPLETED','0:0','256']and states['910374.1535']==['COMPLETED','0:0','1']
 assert watch['completed']and watch['passed']and watch['actual_step']=='910374.1534';post=watch['post'];assert post['passed']and post['KFD_pids']==[]and post['expected_PIDs_still_exist']==[]and post['matching_GPU_driver_PIDs']==[]and len(post['expected_owner_PIDs'])==7
 assert owner['completed']and owner['passed']and owner['planned']==owner['completed_queries']==756 and owner['planned_warm']==126 and owner['failed_queries']==owner['missing_queries']==0 and owner['normal_Runtime_closes']==6 and owner['forced_child_cleanup']==[]
 base=repo/'code/bw1000/Deep-C4-constant-slot-metadata-r482/cache-component-device-v1';manifest=json.loads((base/'GPU_SOURCE_MANIFEST.json').read_bytes());assert H((base/'GPU_SOURCE_MANIFEST.json').read_bytes())=='89454aae76dc234c94340c1ff34c6fe7720a897930990eb8ea060239aac16cf4'
 for n,h in manifest['files_SHA256'].items():assert H((base/n).read_bytes())==h
 pins=json.loads((base/'INPUT_BINDING.json').read_bytes())['input_pins'];assert len(pins)==63
 for p in pins:
  v,s=data(p['layer'],p['epoch']);assert H(v[:p['count']*128])==p['values_SHA256']and H(s[:p['count']*4])==p['scales_SHA256']
 phases=[];raw_verified=0;checksum_outputs=0
 names=['prefix1','prefix2124','prefix8192','reverse8192','page_edges_duplicates','reset_reseed_prefix2124']
 for k,mode in enumerate([0,1,1,0,0,1]):
  q=json.loads((private/('phase%d-report.json'%k)).read_bytes());s=q['schedule'];assert q==owner['phases'][k]['value'];assert q['phase']==k and q['mode']==mode and q['actual_step']=='910374.1534'and q['source_manifest_SHA256']==owner['source_manifest_SHA256']
  assert q['completed']and q['passed']and q['normal_Runtime_close']and q['close']['rc']==0 and q['close']['values'][2:]==[1,21]and q['measured_failed']==0 and not q['Torch_imported']and not q['explicit_Model_or_weights'];assert q['visible_count']==1 and q['physical_PCI']==owner['physical_PCI']
  assert q['symbol_owners']==manifest['symbol_owners'];assert s['completed']and s['passed']and(s['seed_calls'],s['guards_checked'],s['reset_checked'],s['raw_audit_count'])==(63,147,21,252)
  assert s['observation'][2:9]==[21,128,1081344,63,147,147,21];assert s['observation'][9:14]==([1376256,147,105,2446920,42]if mode else[0,147,0,0,147])
  assert len(s['input_pins'])==63
  for p in s['input_pins']:
   pin=[v for v in pins if(v['layer'],v['epoch'],v['count'])==(p['layer'],p['epoch'],p['count'])];assert len(pin)==1 and pin[0]['values_SHA256']==p['values_SHA256']and pin[0]['scales_SHA256']==p['scales_SHA256']
  assert len(s['measured'])==126 and len(s['warm'])==21
  assert[(v['case'],v['layer'])for v in s['measured']]==[(n,l)for n in names for l in range(21)]
  assert[(v['case'],v['layer'])for v in s['warm']]==[('prefix2124',l)for l in range(21)]
  for item in s['measured']+s['warm']:
   idx=indices(item['case']);assert item['count']==len(idx)and H(struct.pack('<'+'q'*len(idx),*idx))==item['IDs_SHA256'];v,scale=gathered(item['layer'],item['case']);assert H(v)==item['values_SHA256']and H(scale)==item['scales_SHA256'];checksum_outputs+=1
   fast=bool(mode and item['case']not in ['reverse8192','page_edges_duplicates']);assert item['counter_delta']==[0,1,int(fast),len(idx)*8 if fast else 0,int(not fast)];assert item['GPU_values_scales_and_CPU_canaries_passed']
   assert len(item['seconds'])==3 and all(math.isfinite(v)and v>=0 for v in item['seconds']);assert item['Python_API_wall_s']>=item['seconds'][2]
  assert len(s['selected_raw'])==8
  for r in s['selected_raw']:
   v,scale=gathered(r['layer'],r['case']);expected=v if r['field']=='values'else scale;b=(private/r['name']).read_bytes();assert b==expected and H(b)==r['SHA256'];raw_verified+=1
  assert len(s['raw_audits'])==252
  for r in s['raw_audits']:
   label=r['label'];epoch=int(label in ['reset_rewritten','after_reset_reseed_prefix2124']);n=2124 if label=='partial_untouched'else 8192
   expect=b'\xa5'*(128*8448)if label=='constructor_A5'else raw_cache(r['layer'],epoch,n)
   assert len(expect)==r['bytes']and H(expect)==r['SHA256'],(k,r['layer'],label)
  hist=json.loads((private/('phase%d-driver.json'%k)).read_bytes());assert len(hist)==q['driver_history']['samples']and len(hist)>1 and q['driver_history']['errors']==[]
  assert all(v[2]==0 and v[3]>=17179869184 and v[4]==68702699520 for v in hist);assert all(b[0]>=a[0]for a,b in zip(hist,hist[1:]));minimum=min(v[3]for v in hist);gap=max(b[0]-a[0]for a,b in zip(hist,hist[1:]));assert minimum==q['driver_min_observed'];close(gap,q['driver_gap_max_s']);assert q['sampled_floor_passed']
  primary=[v for v in s['measured']if v['case']=='prefix2124'];enclosure=sum(v['seconds'][0]for v in primary);api=sum(v['Python_API_wall_s']for v in primary);close(enclosure,s['primary_gather_enclosure_s']);close(api,s['primary_API_wall_s'])
  cases={n:dict(gather_enclosure_s=sum(v['seconds'][0]for v in s['measured']if v['case']==n),output_D2H_s=sum(v['seconds'][1]for v in s['measured']if v['case']==n),probe_before_final_floor_s=sum(v['seconds'][2]for v in s['measured']if v['case']==n),Python_API_wall_s=sum(v['Python_API_wall_s']for v in s['measured']if v['case']==n))for n in names}
  phases.append(dict(phase=k,mode=mode,PID=q['PID'],startticks=q['startticks'],cases=cases,primary_enclosure_s=enclosure,primary_API_wall_s=api,schedule_s=s['elapsed_schedule_s'],worker_s=q['elapsed_worker_s'],driver_min_observed=minimum,driver_max_gap_s=gap,driver_samples=len(hist),constant_table_uploaded_bytes=s['observation'][9],after_create_free=q['after_create'][0],normal_Runtime_close=True))
 assert len({v['PID']for v in phases})==6 and raw_verified==48 and checksum_outputs==882
 pairs=[]
 for a,b in [(0,1),(2,3),(4,5)]:
  off=phases[a]if phases[a]['mode']==0 else phases[b];on=phases[b]if phases[b]['mode']==1 else phases[a]
  pairs.append(dict(phase_order=[a,b],order='AB'if phases[a]['mode']==0 else'BA',ON_over_OFF_gather=on['primary_enclosure_s']/off['primary_enclosure_s'],ON_over_OFF_API_wall=on['primary_API_wall_s']/off['primary_API_wall_s'],ON_minus_OFF_aftercreate_free_bytes=on['after_create_free']-off['after_create_free']))
 return dict(schema='Deep1534-root-three-pair-realcache-v1',actual_GPU='910374.1534',actual_keeper='910407.1118',accounting=rows,post_clean=True,planned_measured=756,completed_measured=756,failed=0,rejected=0,canceled=0,missing=0,warm126_separate=True,independent_input_pins=63,original_checksum_outputs_verified=882,independent_raw_outputs_verified=48,raw_cache_SHA256_audits=1512,normal_Runtime_closes=6,phases=phases,pairs=pairs,primary='sum21 Native gather/status-sync enclosure at past8496 compressedprefix2124, after21warm perprocess; independent process-pair units',median_ON_over_OFF_gather=statistics.median(v['ON_over_OFF_gather']for v in pairs),median_ON_over_OFF_API_wall=statistics.median(v['ON_over_OFF_API_wall']for v in pairs),root_passed=True,Model_GO=False,promote=False,limits=['48 selected raw outputs independently byte replayed; all882 original output hashes reconstructed, producer validated all756measured+126warm actual tensors/canaries.','Rawcache address/state guard/status are producer assertions plus root raw-cache hashes; no independent vendorSDK status replay or fullC4causal/weights/knowledge quality.','Timing primary has3 independent paired process units, not21layers as independent units; Nativegather/statussync enclosure is not exclusiveGPUbusy.','Python APIwall, outputD2H, fullphase worker/loading/hash/audits reported separately; no21/86x extrapolation or Model/clientSLO promise.','All sampled16GiB floors pass, driver5ms target is not actualhardpeak bound; initialDLL/SDKinit and gaps UNKNOWN.','Constanttable extra1.3125MiB/owner logicalpayload; driver/allocator impact and Model fit not inferred frompayload.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=read(a.local,a.repo);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['root_passed','completed_measured','independent_raw_outputs_verified','median_ON_over_OFF_gather','median_ON_over_OFF_API_wall','Model_GO']}))
