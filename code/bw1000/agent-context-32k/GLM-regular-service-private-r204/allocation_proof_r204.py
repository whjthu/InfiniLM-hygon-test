"""Consume exact terminal795 proof once in CPU preparation; no construction."""
import pathlib,hashlib,json
R=pathlib.Path('/refactor');HOST='/public/home/aclsylqidf/hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def mapped(path):
 p=pathlib.Path(path);assert p.is_relative_to(pathlib.Path(HOST));return R/p.relative_to(pathlib.Path(HOST))
def verify():
 from native_chain_r186 import verify as chain
 native=chain();p=R/'GLM-PP4-allocation-r188-watch.json';assert sha(p)=='86fe736cca45afae6223216d301cc8dd5f44a793fa6d1b54876a86643f0f3b9f';q=json.loads(p.read_bytes())
 assert q['qualified']and q['post_empty']and q['actual_handle']=='910376.795'and q['terminal'][2:4]==['COMPLETED','0:0']and not q['weights_loaded']and not q['forward_called']
 assert len(q['owner_results'])==32 and {(x['stage'],x['rank'])for x in q['owner_results']}=={(s,r)for s in range(4)for r in range(8)}
 by_stage={s:[]for s in range(4)}
 for owner in q['owner_results']:
  path=mapped(owner['path']);assert sha(path)==owner['SHA256'];r=json.loads(path.read_bytes());assert r['completed']and r['normal_destroy']and r['normal_shutdown']and r['Torch_postclose_liveness']and r['world32_integer_SUM_exact']and r['all32_constructor_memory_floor_passed']and r['Native_SHA256']==native['library_SHA256']
  assert not r['weights_loaded']and not r['forward_called']and not r['finish_called'];by_stage[r['stage']].append(r['memory_after_trim'][0])
 assert len(q['network_and_normal'])==4 and all(n['normal_markers']==1 and n['IBext_v8_mentions']>=1 and n['Socket_network_mentions']==0 for n in q['network_and_normal'])
 assert len(q['post'])==4 and all(n['KFD_empty']for n in q['post'])
 return {'actual_allocation_handle':'910376.795','watch_SHA256':sha(p),'Native_SHA256':native['library_SHA256'],'CPU543_receipt_SHA256':native['cpu_receipt_SHA256'],'normal_owner_count':32,'normal_node_count':4,'postKFDempty':True,'IBext_all4_noSocket':True,'min_aftertrim_free_bytes_by_stage':[min(by_stage[s])for s in range(4)],'weight_loading_finish_prefill_peak_qualified':False,'source':'actual constructor-only795; never serving or model acceptance'}
