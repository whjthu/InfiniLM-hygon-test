"""No Torch, weights, compiler, network, native load or Runtime construction."""
import ast,hashlib,json,pathlib,re
D=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check_topology(node,rank,tp=32,pp=1):
 if not (0<=node<4 and node*8<=rank<(node+1)*8 and tp==32 and pp==1):raise ValueError('topology')
def run():
 m=json.loads((D/'source-manifest-r120.json').read_bytes())
 for p,h in m['package_source_SHA256'].items():assert sha(D/p)==h,p
 for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
 ranks=[node*8+local for node in range(4) for local in range(8)];assert ranks==list(range(32))
 for node in range(4):
  for local in range(8):check_topology(node,node*8+local)
 for bad in [(-1,0,32,1),(4,32,32,1),(0,8,32,1),(0,0,16,1),(0,0,32,2)]:
  try:check_topology(*bad)
  except ValueError:pass
  else:raise AssertionError('invalid topology accepted')
 covers={}
 for name,global_width,local_width in [('W13rows',3072,96),('W2packedcols',1536,48),('W2scalecols',96,3)]:
  spans=[(rank*local_width,(rank+1)*local_width) for rank in ranks]
  assert [i for a,b in spans for i in range(a,b)]==list(range(global_width));covers[name]=spans
 expert_bytes=896*(2*96*1792+2*96*112+3584*48+3584*3);assert expert_bytes==491323392
 assert expert_bytes*32==15722348544
 cpp=(D/'TP32_I96_MoE_r120.cpp').read_text();assert 'topology.tp_node_size=4' in cpp and 'modern_kimi_set_grouped_prefill(0)' in cpp and 'modern_kimi_set_tiled_w13_r1861(0)' in cpp
 assert not re.search(r'\b(?:setenv|system|dlopen|srun)\s*\(',cpp)
 assert 'for(int arm=0;arm<6;++arm)' in cpp
 assert m['native_SHA256']=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
 return {'passed':True,'CPU_only':True,'AST_files':len(list(D.glob('*.py'))),'rank_coverage':ranks,'shard_coverage':covers,'expert_packed_scale_bytes_per_rank':expert_bytes,'T':[8,16,64],'generic_activation_bytes':{str(T):T*16*96*2 for T in [8,16,64]},'actual_fixture_bytes_read':0,'actual_distinct_fixture_rows':'PENDING_GPUconsumer_CPU_fixture_read; no16distinct claim','compile':False,'GPU':False,'native_load':False}
if __name__=='__main__':print(json.dumps(run(),indent=2))
