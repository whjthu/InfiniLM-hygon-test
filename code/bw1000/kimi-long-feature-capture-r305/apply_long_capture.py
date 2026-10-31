"""Apply an exact isolated HOST-only feature capture patch to pinned247 source."""
from pathlib import Path
import hashlib,json,shutil,sys
D=Path(__file__).resolve().parent
PINS={'batch_live_taps_r9122.hpp':'6aaa5200e4bafd55282f193449b8718488a5e3e4de86dc30b5da07a0a2a87877','pair_probe.cpp':'38dbbe54e90d4150ee510acac774254b9e9e1ff5da36d210e8e36eebf84240ce'}
INDEX_SHA='b891594064ecc089f214d4aa2cf07922bef9f66002c705c53ad076ef084e3c5e'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def patch(src,dst,index_path=None):
 src=Path(src);dst=Path(dst);assert not dst.exists();index_path=Path(index_path)if index_path else src.parent/'native-source-index.json';assert sha(index_path)==INDEX_SHA;index=json.loads(index_path.read_bytes());assert len(index)==22
 for n,h in PINS.items():assert sha(src/n)==h
 for n,h in index.items():assert sha(src.parent/n)==h
 dst.mkdir(parents=True)
 for n in index:
  p=dst/Path(n).relative_to('native');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((src.parent/n).read_bytes())
 p=dst/'batch_live_taps_r9122.hpp';old=p.read_text();s=old
 def one(a,b):
  nonlocal s
  assert s.count(a)==1,a;s=s.replace(a,b)
 one('#include <cstdint>','#include <cstdint>\n#include "long_capture_geometry_r305.hpp"\n#include "long_capture_api_r305.hpp"')
 one('size_t total=0;};','size_t total=0;size_t context_limit=1024,pages_per_row=4;bool scoped_long=false;};')
 one('auto past=fc.attn_metadata.past_sequence_lengths','const auto capture_scope=expected.scoped_long?kimi_long_features_r305::long_scope():kimi_long_features_r305::legacy_scope();if(expected.context_limit!=capture_scope.context||expected.pages_per_row!=capture_scope.pages)throw std::logic_error("r305 stale capture geometry");\n auto past=fc.attn_metadata.past_sequence_lengths')
 one('Shape{C,4}','Shape{C,expected.pages_per_row}')
 one('bb(C*4)','bb(C*expected.pages_per_row)')
 one('blocks->data(),C*16','blocks->data(),bb.size()*sizeof(int32_t)')
 start=s.index(' for(size_t i=0;i<C;++i){if(pp[i]!=expected.past[i]');end=s.index('\n auto contiguous=',start)
 s=s[:start]+' kimi_long_features_r305::validate_observed(capture_scope,expected.epoch,C,expected.logical.data(),expected.past.data(),expected.width.data(),pp.data(),tt.data(),oo.data(),bb.data());'+s[end:]
 one('size_t cursor=0;','')
 marker='}\nextern "C" int modern_kimi_batch_taps_begin_r9122';assert s.count(marker)==1;s=s.replace(marker,(D/'long_capture_arm_r305.inc').read_text()+marker)
 # Existing public begin/count/info/copy/capabilities implementation stays exact.
 suffix=old[old.index('extern "C" int modern_kimi_batch_taps_begin_r9122'):];assert s.endswith(suffix)
 assert 'Shape{C,4}'not in s and 'bb(C*4)'not in s and 'blocks->data(),C*16'not in s
 p.write_bytes(s.encode());p=dst/'pair_probe.cpp';p.write_bytes(p.read_bytes()+b'\n'+(D/'long_capture_wrapper_r305.inc').read_bytes())
 for n in ['long_capture_geometry_r305.hpp','long_capture_api_r305.hpp']:(dst/n).write_bytes((D/n).read_bytes())
 unchanged={n:h for n,h in index.items()if Path(n).name not in PINS};assert all(sha(dst/Path(n).relative_to('native'))==h for n,h in unchanged.items())
 return {'old_pins':PINS,'new_SHA256':{n:sha(dst/n)for n in [*PINS,'long_capture_geometry_r305.hpp','long_capture_api_r305.hpp']},'all_other_source_members_unchanged':len(unchanged),'legacy_public_capture_functions_byte_exact':True,'pair_source_only_appended_independent_CAPI':True,'new_HOST_TUs':2,'new_GPU_TUs':0,'full_Native_linked':False,'GPU_GO':False}
if __name__=='__main__':print(json.dumps(patch(sys.argv[1],sys.argv[2],sys.argv[3]if len(sys.argv)>3 else None),indent=2))
