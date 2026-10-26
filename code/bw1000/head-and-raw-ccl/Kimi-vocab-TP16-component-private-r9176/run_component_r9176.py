"""Default-OFF TP16 head-only real checkpoint component. No complete Model load."""
import pathlib,os,json,hashlib,ctypes as C,time,socket
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def layout_contract(M,V=163840,parts=16):
 assert M in [1,4,8,16,32,64]and V%parts==0;return {'local_shape':[M,V//parts],'gather_rank_major_shape':[parts*M,V//parts],'output_shape':[M,V],'local_weight_shape':[V//parts,7168],'local_weight_bytes':V//parts*7168*2}
def port_counters():
 values={}
 for root in pathlib.Path('/sys/class/infiniband').glob('*/ports/1/counters'):
  for name in ['port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets']:
   p=root/name
   try:values[str(p)]=int(p.read_bytes().strip())
   except(OSError,ValueError):pass
 return {'raw':values,'data_counter_units':'IB port data counters are 4-byte units; aggregate port traffic, not isolated head wirebytes','GDR':'UNKNOWN'}
def run():
 assert os.environ.get('KIMI_VOCAB_COMPONENT_R9176')=='1','defaultOFF';m=json.loads((D/'manifest-r9176.json').read_bytes());b=json.loads(pathlib.Path(m['CPU_receipt']).read_bytes());assert b['completed']and not b['GPU']and not b['dlopen']and sha(b['native'])==m['native_SHA256']==b['native_SHA256']
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 for p,h in b['all_linked_object_SHA256'].items():assert sha(p)==h,p
 model=pathlib.Path(m['model_view']);assert sha(model/'config.json')==m['config_SHA256']and sha(model/'model.safetensors.index.json')==m['index_SHA256'];node=int(os.environ['SLURM_PROCID']);assert socket.gethostname()==m['nodes'][node]and len(os.sched_getaffinity(0))>=64 and int(os.environ['SLURM_CPUS_PER_TASK'])==64
 if os.environ.get('KIMI_VOCAB_CPU_PREFLIGHT_R9176')=='1':
  import sys
  assert 'torch'not in sys.modules;print(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'Runtime_created':False,'passed':True,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':node,'host':socket.gethostname(),'affinity':len(os.sched_getaffinity(0)),'manifest_SHA256':sha(D/'manifest-r9176.json')}),flush=True);return
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;mode=os.environ['KIMI_VOCAB_MODE_R9176'];assert mode in ['dual','full','shard'];tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi-vocab-TP16-r9176-'+tag+'-node%d-'%node+mode+'.json');assert not out.exists();start=time.perf_counter();index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];key='language_model.lm_head.weight'
 with safe_open(model/index[key],framework='pt',device='cpu')as f:weight=f.get_tensor(key).contiguous()
 assert weight.device.type=='cpu'and weight.dtype==torch.bfloat16 and tuple(weight.shape)==(163840,7168);load=time.perf_counter()-start;generator=torch.Generator().manual_seed(917600);arrays=[];trials=[]
 for M in [1,4,8,16,32,64]:
  x=(torch.randn((M,7168),generator=generator)*.02).to(torch.bfloat16).contiguous();arrays.append(x);trials.append({'rows':M,'input_pointer':x.data_ptr(),'CPU_input_SHA256':hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest()});layout_contract(M)
 before=port_counters();lib=C.CDLL(b['native']);lib.modern_kimi_vocab_TP16_component_r9176.argtypes=[C.c_char_p];lib.modern_kimi_vocab_TP16_component_r9176.restype=C.c_int;lib.modern_kimi_vocab_TP16_result_r9176.argtypes=[];lib.modern_kimi_vocab_TP16_result_r9176.restype=C.c_char_p
 payload={'node':node,'mode':mode,'head':os.environ['KIMI_VOCAB_HEAD_ADDR_R9176'],'port':int(os.environ['KIMI_VOCAB_BASEPORT_R9176'])+['dual','full','shard'].index(mode),'weight_pointer':weight.data_ptr(),'trials':trials};tick=time.perf_counter();rc=lib.modern_kimi_vocab_TP16_component_r9176(json.dumps(payload).encode());ended=time.perf_counter();raw=lib.modern_kimi_vocab_TP16_result_r9176();assert raw;answer=json.loads(raw);answer.update(actual_step=tag,node=node,host=socket.gethostname(),manifest_SHA256=sha(D/'manifest-r9176.json'),DSO_SHA256=sha(b['native']),checkpoint_key=key,checkpoint_shard=index[key],CPU_checkpoint_read_s=load,head_component_wall_s=ended-tick,hidden_provenance='seed917600 synthetic BF16, identical CPUgenerator bothnodes; not Model/taskquality',input_SHA256=[x['CPU_input_SHA256']for x in trials],network_before=before,network_after=port_counters(),GPU_peak_scope='C++ synchronous free/total at constructor/load/warm/M64-with-output; exact transient peak and Runtime live/reserved counters UNKNOWN',all16_final_reports_need_bothnodes=True,actual_CPU_affinity_count=len(os.sched_getaffinity(0)))
 out.write_bytes(json.dumps(answer,indent=2).encode('utf-8'));assert rc==0 and answer['completed']and answer['communication_group_destroyed']and answer['all8_localowners_closed'];assert len(answer['reports'])==8 and all(x['Runtime_owner_closed']and x['heads_destroyed_before_Runtime']for x in answer['reports'])
 for rep in answer['reports']:
  for k in ['memory_before_heads','memory_after_constructor','memory_after_load','memory_after_head_destroy_trim']:assert rep[k]['free']>4*1024**3,('4GiB component observedsample reserve',k,rep[k])
  for case in rep['cases']:
   if 'memory_M64_with_full_output_held'in case:assert case['memory_M64_with_full_output_held']['free']>4*1024**3
 if mode=='dual':assert all(c['numerics']['all_finite']and c['numerics']['normalized_L2']<=.02 for r in answer['reports']for c in r['cases']),'unexpected large component numeric error; NOT tokenidentity gate'
 print('KIMI_VOCAB_NODE_R9176_NORMAL',node,mode,flush=True)
if __name__=='__main__':run()
