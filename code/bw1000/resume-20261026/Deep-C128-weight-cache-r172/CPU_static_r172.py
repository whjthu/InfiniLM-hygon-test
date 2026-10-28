"""Pure file/source closure checks. No import of Torch/Native or device initialization."""
import pathlib,json,hashlib,re
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check():
 m=json.loads((D/'manifest-r172.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 c=json.loads((D/'source-context-r172.json').read_bytes())
 assert c['current_CPU_receipt_SHA256']=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c'
 assert len(c['current_link']['objects'])==48 and len(c['units'])==9
 src=(D/'src/deepseek_v4_compressor_fp32_c128_r19.cpp').read_text()
 hdr=(D/'src/deepseek_v4_temporal_compressor.hpp').read_text()
 assert 'std::atomic<int> policy{0};'in src and 'thread_local'not in src
 assert 'runtime::Tensor wkv32_r172_, wgate32_r172_;'in hdr
 assert src.count('cached_generation_r172_=++weight_generation_r172_;')==1
 finish=src.split('void TemporalCompressor::process_weights_after_loading',1)[1].split('void TemporalCompressor::rotate',1)[0]
 assert 'if(ready_)throw' in finish
 assert finish.index('r.syncStream();\n        } catch')<finish.index('wkv32_r172_=std::move(kv32)')<finish.index('ready_=true')
 assert 'wkv_,kv32' in finish and 'wgate_,gate32' in finish
 prepare=src.split('TemporalCompressor::Batch TemporalCompressor::prepare_impl',1)[1].split('void TemporalCompressor::commit',1)[0]
 assert prepare.index('check_owner(r);')<prepare.index('cache_on_r172')
 assert 'cached_generation_r172_!=weight_generation_r172_'in prepare
 assert 'cache_on_r172?cached32:work.alloc' in prepare and 'if(!cache_on_r172)'in prepare
 assert 'hidden32,weight32->permute({1,0})'in prepare and 'out32,out' in prepare
 assert src.count('projection(wkv_,0,wkv32_r172_)')==1
 assert src.count('projection(wgate_,channels_/2*dim_,wgate32_r172_)')==1
 # Unmodified source copies are exact normalized semantic copies of ac937 bindings.
 for n,h in c['original_source_text_SHA256'].items():
  if n not in ['deepseek_v4_compressor_fp32_c128_r19.cpp','deepseek_v4_temporal_compressor.hpp']:
   assert sha(D/'src'/n)==h,n
 # Verify all recorded actual C1 pasts are component obligations, not substitute B32.
 plan=json.loads((D/'component-obligations-r172.json').read_bytes())
 assert plan['actual_C1_pasts']==[385,395,437,407]
 assert plan['smallT_mode']==0 and plan['tokens']==plan['requests']==1
 return dict(passed=True,scope='SOURCE_ONLY; no actual Native lifecycle/GPU/numerical qualification',source_manifest_SHA256=sha(D/'manifest-r172.json'),HOST_TUs_planned=9,GPU_TUs_planned=0,new_cache_bytes_per_owner=335544320)
if __name__=='__main__':print(json.dumps(check()))
