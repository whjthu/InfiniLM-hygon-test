"""Pure source/profile consumption; never import/dlopen Native or SDK."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent;N=D/'native'
sha=lambda b:hashlib.sha256(b).hexdigest()
def body(s,marker):
 i=s.index(marker);i=s.index('{',i);depth=1;j=i+1
 # Retained functions contain no braces inside string literals except their
 # literal error text, which is brace-free. This is checked on exact sources.
 while depth:
  depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[i:j]
def check(originals):
 get=lambda p:originals[p].replace('\r\n','\n')
 pair='/refactor/incoming/Kimi-append512-r247/native/pair_probe.cpp'
 old=get(pair);new=(N/'pair_probe.cpp').read_text();proof={}
 assert body(old,'struct Context')==body(new,'struct Context')
 proof['Context_definition_unchanged']=sha(body(old,'struct Context').encode())
 for marker in ['int input_forward_locked_r247(','modern_kimi_engine_create_long_agent_r179(','modern_kimi_engine_create_batch_r9122c(','modern_kimi_pair_append_true_input_r247(','modern_kimi_engine_reset_r2475(','modern_kimi_engine_destroy_r2475(']:
  assert body(old,marker)==body(new,marker),marker
  proof[marker]=sha(body(old,marker).encode())
 # Explicit typed API, no alternate flag bypass; normal reset/destroy reused.
 assert 'geometry_r453!=authorized_long_geometry_r453'in new
 assert '"INFINILM_KIMI_LONG_GEOMETRY_R453"'in new
 assert 'if(!c->engine->append_ready_r247())'in new
 assert 'c->pair_r9006||c->speculation_r3632||c->batch_enabled_r9122||c->batch_capture_r9122'in new
 for marker in ['modern_kimi_engine_forward_r2475(','modern_kimi_engine_resize_cache_r2475(','modern_kimi_engine_set_cache_pages_r4974(','modern_kimi_long_agent_capabilities_r179(']:
  assert 'geometry_context_r453(c)'in body(new,marker)
 scale='/refactor/incoming/Kimi-long-scale-exact-r201/'
 old=get(scale+'scale_exact_scope_r201.hpp');new=(N/'scale_exact_scope_r201.hpp').read_text()
 assert body(old,'struct Geometry')==body(new,'struct Geometry')
 assert 'Ticket ticket_;bool armed_=false;'in old and 'Ticket ticket_;bool armed_=false;'in new
 assert body(old,'~Scope()')==body(new,'~Scope()')
 old=get(scale+'kimi_moe_scale_exact_r201.cpp');new=(N/'kimi_moe_scale_exact_r201.cpp').read_text()
 assert new.replace('&runtime,this,model_config->get_config_json().value("infinilm_kimi_long_geometry_optin_r453",false));','&runtime,this);')==old
 head='/refactor/incoming/Kimi-shard-only-Model-r9183/'
 old=get(head+'kimi_shard_head_r9183.hpp');new=(N/'kimi_shard_head_r9183.hpp').read_text()
 assert old[old.rindex('private:'):]==new[new.rindex('private:'):]
 marker='infinilm::runtime::Tensor forward('
 assert body(old,marker)==body(new,marker)
 proof['head_forward_math_and_fence_unchanged']=sha(body(old,marker).encode())
 for name in ['kimi_shard_model_r9183.cpp','kimi_shard_model_r9183.hpp','kimi_shard_policy_r9183.cpp']:
  assert (N/name).read_text()==get(head+name),name
 assert (N/'long_geometry_r453.hpp').read_bytes()==(D/'long_geometry_r453.hpp').read_bytes()
 from profile import PROFILES,select,LIMITS,CREATE_ARGTYPES,FORWARD_ARGTYPES
 assert len(CREATE_ARGTYPES)==6 and len(FORWARD_ARGTYPES)==14
 assert LIMITS['precreate_allowner_floor_bytes']==4*1024**3 and LIMITS['context']==32768
 ranks={};cases=0
 for name,p in PROFILES.items():
  assert select(name,optin=True)==p
  try:select(name);raise AssertionError('default was enabled')
  except ValueError:pass
  world=[];emitter=set()
  for n in range(4):
   args=p.create_arguments(b'{"recording":true}',n,b'12.1.2.1',43872,{'INFINILM_KIMI_LONG_GEOMETRY_R453':'1'})
   assert args[1:4]==(n,p.tp_nodes,p.pp)
   for d in range(8):
    r=p.rank(n,d);world.append(r['world_rank']);cases+=1
    assert r['local_MLA_heads']*p.tp==96 and r['local_expert_I']*p.tp==3072
    if r['emitter_node']:emitter.add(n)
  assert sorted(world)==list(range(32)) and emitter=={p.emitter_node}
  ranks[name]=dict(world=world,emitter_node=p.emitter_node)
 try:select('unknown',optin=True);raise AssertionError('unknown admitted')
 except ValueError:pass
 return dict(passed=True,pure_source=True,Native_dlopen=False,Runtime_created=False,GPU=0,source_proof=proof,recorded_create_rank_cases=cases,profiles=ranks)
def originals_remote():
 d=json.loads((D/'DERIVATION.json').read_bytes());out={}
 for p,h in d['original_raw_source_SHA256'].items():
  b=Path(p).read_bytes();assert sha(b)==h,p;out[p]=b.decode()
 return out
if __name__=='__main__':
 if len(sys.argv)>1:
  q=json.loads(Path(sys.argv[1]).read_bytes());o={v['path'].replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor'):v['rawtext']for v in q['source_files']}
 else:o=originals_remote()
 print(json.dumps(check(o),indent=2))
