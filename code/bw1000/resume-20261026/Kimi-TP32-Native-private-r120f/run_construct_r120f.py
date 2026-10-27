"""Future bounded GPU entry only, no submit code. No weights or model forward."""
import pathlib,json,hashlib,os,socket,datetime,re,sys
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(path):
 D=pathlib.Path(__file__).resolve().parent;m=json.loads(pathlib.Path(path).read_bytes());s=json.loads((D/'manifest-r120f.json').read_bytes())
 assert m['explicit_constructor_GO']is True and m['root_source_published']is True and re.fullmatch('[0-9a-f]{40}',m['publication_commit'])
 assert m['source_manifest_SHA256']==sha(D/'manifest-r120f.json')
 for n,h in s['source_SHA256'].items():assert sha(D/n)==h,n
 for p,h in s['required_DSO_SHA256'].items():assert sha(p)==h,p
 nodes=['a02r4n08','a02r4n09','a02r4n10','a02r4n11'];assert m['nodes']==nodes and m['parent_allocation']=='910375'
 node=int(os.environ['SLURM_PROCID']);assert os.environ['SLURM_JOB_ID']=='910375'and 0<=node<4 and socket.gethostname()==nodes[node]and len(os.sched_getaffinity(0))>=32
 fresh=m['fresh_owned32_before'];assert fresh['qualified']is True and fresh['KFD_empty_all4']is True and fresh['nodes']==nodes and fresh['port_free_all4']is True
 assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(fresh['UTC'])).total_seconds()<=120
 assert m['master_addr'].startswith('12.')and 1024<int(m['master_port'])<65535
 cp=pathlib.Path('/refactor/candidates/Kimi-TP32-Native-r120f/CPU-build.json');wp=pathlib.Path('/refactor/candidates/Kimi-TP32-construct-r120f/CPU-build.json');c=json.loads(cp.read_bytes());w=json.loads(wp.read_bytes())
 assert sha(cp)==m['Native_CPU_receipt_SHA256']and sha(wp)==m['wrapper_CPU_receipt_SHA256']and c['completed']and w['completed']and c['CPU_only']and w['CPU_only']and not c['GPU']and not w['GPU']and not c['dlopen']and not w['dlopen']
 assert c['source_manifest_SHA256']==m['source_manifest_SHA256']and w['parent_CPU_receipt_SHA256']==sha(cp)
 assert c['HOST_TUs_compiled']==5 and w['HOST_TUs_compiled']==1 and c['GPU_TUs_compiled']==w['GPU_TUs_compiled']==0
 assert sha(c['native'])==c['native_SHA256']==w['native_SHA256']and sha(w['library'])==w['library_SHA256']==m['wrapper_SHA256']
 for p,h in c['all_linked_object_SHA256'].items():assert sha(p)==h,p
 cfg=pathlib.Path('/shared/Kimi-K3/config.json');assert sha(cfg)==m['config_SHA256']=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
 # All authorization/source/CPU/fresh-resource gates precede Torch/DSO loading.
 import torch,ctypes
 assert torch.cuda.device_count()==8;torch.set_num_threads(1)
 lib=ctypes.CDLL(w['library']);lib.modern_kimi_TP32_construct_r120f.argtypes=[ctypes.c_char_p];lib.modern_kimi_TP32_construct_r120f.restype=ctypes.c_int;lib.modern_kimi_TP32_construct_result_r120f.restype=ctypes.c_char_p
 out=pathlib.Path('/refactor/results')/('Kimi-TP32-construct-r120f-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+str(node)+'.json');assert not out.exists()
 rc=lib.modern_kimi_TP32_construct_r120f(json.dumps({'node_rank':node,'master_addr':m['master_addr'],'master_port':m['master_port'],'config':json.loads(cfg.read_bytes())}).encode());q=json.loads(lib.modern_kimi_TP32_construct_result_r120f());q.update(node_rank=node,returncode=rc,CPU_receipt_SHA256=sha(cp),wrapper_CPU_receipt_SHA256=sha(wp),wrapper_SHA256=w['library_SHA256'],Native_SHA256=c['native_SHA256'],runtime_manifest_SHA256=sha(path),source_manifest_SHA256=m['source_manifest_SHA256'],publication_commit=m['publication_commit'])
 with out.open('x')as f:json.dump(q,f,indent=2)
 assert rc==0 and q['completed']and q['all8normal_before_communicators']and not q['full93_model_parameters_constructed']and q['forward_calls']==0
 assert sorted(x['global_rank']for x in q['owners'])==list(range(node*8,(node+1)*8))and all(x['normal_Runtime_close']and x['real32_BF16_headwidth_allgather']for x in q['owners'])
 print('KIMI_TP32_CONSTRUCT_R120F_NORMAL',node,flush=True)
if __name__=='__main__':run(sys.argv[1])
