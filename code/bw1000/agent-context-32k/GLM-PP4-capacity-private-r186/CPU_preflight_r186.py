"""Original-server CPU metadata and local source-contract only. No dlopen/GPU."""
import ast,hashlib,json,pathlib,os,re
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def validate(cpu_receipt_path=None,cpu_receipt_sha=None):
 from native_chain_r186 import verify,CPU,CPU_SHA
 native=verify(CPU if cpu_receipt_path is None else cpu_receipt_path,CPU_SHA if cpu_receipt_sha is None else cpu_receipt_sha)
 manifest=json.loads((D/'source-manifest-r186.json').read_bytes())
 for n,h in manifest['files'].items():assert sha(D/n)==h,n
 from preflight_r133 import validate as parent
 p=parent()
 cp=pathlib.Path('/refactor/candidates/GLM-critical-observer-r131/CPU-build.json');proof=json.loads(cp.read_bytes())
 observed=[]
 for x in json.loads((D/'source-pins-r186.json').read_bytes()):
  s=pathlib.Path(x['remote']);assert sha(s)==x['SHA256'],x['component']
  q={'component':x['component'],'path':str(s),'SHA256':sha(s),'bytes':s.stat().st_size}
  if 'object'in x:
   assert sha(pathlib.Path(x['object']))==x['object_SHA256']
   q['object_SHA256']=x['object_SHA256'];q['MD_SHA256']=x['MD_SHA256']
  observed.append(q)
 cfgpath=pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json');cfg=json.loads(cfgpath.read_bytes())
 assert cfg['num_hidden_layers']==78 and cfg['hidden_size']==6144 and cfg['max_position_embeddings']>=32768
 bounds=[0,18,38,58,78];counts=[]
 for i,(a,b)in enumerate(zip(bounds,bounds[1:])):
  assert cfg['indexer_types'][a]=='full'
  sparse=sum(x=='sparse' for x in cfg['mlp_layer_types'][a:b])
  full=sum(x=='full' for x in cfg['indexer_types'][a:b])
  counts.append({'stage':i,'begin':a,'end':b,'layers':b-a,'sparse_layers':sparse,'full_indexer_layers':full,'projected_KV_bytes':2*32768*8*256*2*(b-a),'latent_rotary_bytes':32768*(512+64)*2*(b-a),'indexer_cache_bytes':32768*128*2*full})
 tree=ast.parse((D/'allocation_entry_r186.py').read_text())
 forbidden={'modern_glm_model_load_batch','modern_glm_model_finish','modern_glm_model_forward','modern_glm_model_resident_mask_r5640','modern_glm_model_transient_policy_r5688','modern_glm_model_owned_prefix_policy_r1669','modern_glm_model_feature_bypass_r1406'}
 calls={n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
 assert not calls&forbidden,calls&forbidden
 assert "GPU_authorized"in(D/'allocation_entry_r186.py').read_text()
 return {'completed':True,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'weights_read':False,'Native_forward_called':False,'source_manifest_SHA256':sha(D/'source-manifest-r186.json'),'parent_metadata':native,'actual_sources':observed,'config_SHA256':sha(cfgpath),'max_position_embeddings':cfg['max_position_embeddings'],'TP':8,'PP':4,'rows':1,'capacity':32768,'bounds':bounds,'stage_cache_logical':counts,'known_forward_call_limit':4096,'constructor_allocates_parameter_storage':True,'constructor_fit':None,'finish_peak':None,'long_prefill_workspace_peak':None,'model_agent_qualified':False,'normal_agent_target':'32K/4K output,8K thinking,8 actual tools,8K then16K+ real input; CPU sizing is not acceptance'}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-receipt-path',required=True);ap.add_argument('--cpu-receipt-sha',required=True);a=ap.parse_args()
 q=validate(pathlib.Path(a.cpu_receipt_path),a.cpu_receipt_sha);out=D/'CPU-result-r186.json';assert not out.exists();out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
