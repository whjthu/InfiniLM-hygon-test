"""Inspect and CPU-execute actual Kimi metadata guards, without any GPU runtime."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,threading
G=Path(__file__).resolve().parents[1];N=G/'local-only/Kimi-append512-host-candidate-r247/native';R=G/'local-only/Kimi-append512-service-r252/runtime';DR=G/'code/bw1000/agent-context-32k/Kimi-regular-agent32K-private-r189b/runtime'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run():
 taps=N/'batch_live_taps_r9122.hpp';assert sha(taps)=='6aaa5200e4bafd55282f193449b8718488a5e3e4de86dc30b5da07a0a2a87877'
 pair=N/'kimi_pair_state_r9006.hpp';assert sha(pair)=='e60dbf05205fc143833c68de4d865da185e6839e8bb28bcd75df42c338091065'
 text=taps.read_text();begin=text.index('extern "C" int modern_kimi_batch_taps_begin_r9122(');start=text.index('{',begin);depth=1;i=start+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 function=text[begin:i];declarations=text[text.index('namespace kimi_batch_live_r9122 {'):text.index('bool representative(')]+'}\n'
 assert 'blocks->shape()!=infinilm::runtime::Shape{C,4}'in text and 'bb(C*4)'in text and 'blocks->data(),C*16'in text
 program=r'''
#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include <set>
#include <array>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include "kimi_pair_state_r9006.hpp"
'''+declarations+function+r'''
int main(){
 int owner=0,logical=1;int32_t past=0,width=512;int positive=0,negative=0;
 auto check=[&](bool yes){if(!yes)throw std::runtime_error("CPU predicate failed");};
 for(auto p:{0,512,1023}){past=p;width=p==0?512:1;check(modern_kimi_batch_taps_begin_r9122(&owner,1,1,1,&logical,&past,&width)==0);++positive;}
 for(auto p:{1024,8635,32760}){past=p;width=8;check(modern_kimi_batch_taps_begin_r9122(&owner,2,1,1,&logical,&past,&width)!=0);++negative;}
 check(modern_kimi_batch_taps_begin_r9122(&owner,2,0,0,nullptr,nullptr,nullptr)==0);++positive;
 for(int endpast:{8635,32760})for(size_t accepted=0;accepted<8;++accepted){
  kimi_pair_r9006::State state(&owner,1,true);check(state.geometry.logical==1&&state.geometry.physical==2&&state.geometry.half==1&&state.geometry.context==32768&&state.geometry.state_rows()==3&&state.geometry.pages()==260);state.prefilled(&owner,1,512);
  while(state.row(&owner,1).past<endpast){size_t n=std::min(size_t(8),size_t(endpast-state.row(&owner,1).past));std::vector<int64_t>v(n,42);state.completed(&owner,1,v);state.commit(&owner,1,n-1,true);}
  auto old=state.row(&owner,1);std::vector<int64_t>v{11,12,13,14,15,16,17,18};state.completed(&owner,1,v);state.commit(&owner,1,accepted,true);
  if(accepted<7){check(state.row(&owner,1).phase==3&&state.row(&owner,1).past==endpast);auto wrong=std::vector<int64_t>(v.begin(),v.begin()+accepted+1);wrong[0]=999;bool rejected=false;try{state.check(&owner,1,endpast,old.active,old.shadow,wrong);}catch(const std::invalid_argument&){rejected=true;}check(rejected);++negative;std::vector<int64_t>kept(v.begin(),v.begin()+accepted+1);state.completed(&owner,1,kept);}
  auto now=state.row(&owner,1);check(now.phase==1&&now.past==endpast+int(accepted+1)&&now.active==old.shadow&&now.shadow==old.active);++positive;
 }
 std::cout<<"{\"passed\":true,\"positive\":"<<positive<<",\"negative\":"<<negative<<",\"actual_legacy_tap_begin_rejected_past\":[1024,8635,32760],\"actual_long_pair_prefix_lengths_tested\":[8635,32760],\"accepted_prefixes_each\":8}";
}
'''
 compiler=shutil.which('g++');assert compiler
 with tempfile.TemporaryDirectory(prefix='kimi-long-spec-reuse-')as td:
  d=Path(td);(d/'kimi_pair_state_r9006.hpp').write_bytes(pair.read_bytes());p=d/'guard.cpp';p.write_bytes(program.encode());exe=d/('guard.exe'if os.name=='nt'else'guard');argv=[compiler,'-std=c++17','-O2','-pthread',str(p),'-o',str(exe)]
  build=subprocess.run(argv,text=True,capture_output=True,timeout=45,cwd=d);assert build.returncode==0,build.stderr[-2000:]
  z=subprocess.run([str(exe)],text=True,capture_output=True,timeout=15,cwd=d);assert z.returncode==0,z.stderr;actual=json.loads(z.stdout);compiler_version=subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=10).stdout.splitlines()[0]
 # Execute only CPU cache metadata classes; no Torch import or tensor allocation.
 sys.path.insert(0,str(DR));ctx=load('longspec_actual_context',DR/'draft_context_r9128.py');committed=load('longspec_actual_committed',DR/'committed_context_r9202.py')
 raw=ctx.ContextStore(1);raw.begin(1);raw.append(1,1,0,512,('seed',),lambda a,b:a+b);raw.append(1,1,512,512,('next',),lambda a,b:a+b)
 try:raw.validate_many(1,[{'request':1,'past':1024,'rows':8}])
 except AssertionError:raw_rejected=True
 else:raise AssertionError('Expected raw cache1024 limit')
 c=committed.CommittedFeatureContexts.__new__(committed.CommittedFeatureContexts);c.owner=threading.get_ident();c.raw=ctx.ContextStore(1);c.raw.begin(1);c.raw.append(1,1,0,512,('seed',),lambda a,b:a+b);c.enabled=False
 try:c.validate_many(1,[{'request':1,'past':512,'rows':512}])
 except AssertionError:chunk_rejected=True
 else:raise AssertionError('Expected after-prefill committed chunk<=8 bound')
 sys.path.pop(0);assert 'torch'not in sys.modules
 sources=[taps,pair,N/'batch_state_r9122.hpp',N/'pair_probe.cpp',N/'worker.cpp',R/'long_native_r179.py',R/'native_append_r248.py',R/'long_ar_append_r248.py',R/'batch_cohort_r9128.py',R/'batched_query_r9142.py',DR/'draft_context_r9128.py',DR/'committed_context_r9202.py',DR/'query_projected_r9200.py']
 return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'ACTUAL_CPU_GUARDS_LOCALIZED_32K_SPECULATIVE_REUSE_BLOCKERS','CPU_only':True,'GPU':0,'Native_dlopen':0,'Torch_import':False,'weights':0,'HTTP':0,'compiler':compiler_version,'actual_CPP_guard_and_pair_state':actual,'actual_legacy_tap_begin_fragment_SHA256':hashlib.sha256(function.encode()).hexdigest(),'compiled_fragment_scope':'Exact extracted legacy taps_begin plus actual pure State header; not whole Native/trace HOST TU compile or GPU semantics','static_collect_guard':'Actual header still requires block_tables[C,4] and C*16 copy bytes, while long allocator/CAPI uses128pages','actual_raw_ContextStore_rejects_1025':raw_rejected,'actual_committed_ContextStore_rejects_second512_prefill_chunk':chunk_rejected,'source_paths_SHA256':{p.relative_to(G).as_posix():sha(p)for p in sources},'snapshot_scope':'Pinned247 target Native host snapshot plus252 AR service and historical189b draft reuse files; no assertion of an executing draft service','current_AR_affected':False,'new_speedup_claim':False,'Model_qualified':False,'next':'Keep old Native/AR untouched; preserve exact capture/cache/query blockers. A separate optional defaultOFF long capture CPU card must validate real Context geometry before arming, then actual HOST compile without linking/dlopen; prioritize pending actual C1T1 profile over blind draftcache expansion.'}
if __name__=='__main__':
 q=run();p=Path(sys.argv[1]);assert not p.exists();p.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(q,ensure_ascii=False,indent=2))
