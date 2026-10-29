"""File-only adaptation contract; not a native build or Model qualification."""
import ast,pathlib,json,hashlib,sys
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check():
 m=json.loads((D/'manifest-r216.json').read_bytes())
 for p,h in m['source_SHA256'].items():assert sha(D/p)==h,p
 for p in D.rglob('*.py'):ast.parse(p.read_bytes())
 c=json.loads((D/'source-context-r216.json').read_bytes());old=json.loads((D/'source-context-r172.json').read_bytes());a=json.loads((D/'adaptation-r216.json').read_bytes())
 assert c['parent_receipt_SHA256']=='17e94923049bfe910a3cb4810fb69d9c9e4bdebc9265ad2c21d5eaa65c2d39ac'and c['parent_library_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f'
 assert len(old['units'])==len(a['nine_HOST_rebuilds'])==9 and len(c['actual_parent_reused'])==45
 assert len(c['actual_parent_compiled'])==3 and a['nine_HOST_rebuilds'][0]['current232_object'].endswith('/Deep-native-context-guard-r180d/0.o')
 assert sha(D/'include/native_sparse_cache_r22.hpp')==c['current_guard_SHA256']and b'capacity>65536'in(D/'include/native_sparse_cache_r22.hpp').read_bytes()
 src=(D/'src/deepseek_v4_compressor_fp32_c128_r19.cpp').read_text();h=(D/'src/deepseek_v4_temporal_compressor.hpp').read_text()
 assert 'std::atomic<int> policy{0};'in src and 'thread_local'not in src and 'runtime::Tensor wkv32_r172_, wgate32_r172_;'in h
 assert 'cached_generation_r172_=++weight_generation_r172_;'in src and 'cached_generation_r172_!=weight_generation_r172_'in src
 assert a['cache_bytes']['per_owner']==2*512*4096*4*20==335544320
 assert all('torch'not in sys.modules for _ in [0])and '_infinilm'not in sys.modules
 return {'passed':True,'source_only':True,'CPU_build':'NOT_RUN','nine_planned_HOST':9,'HIP':0,'parent_Native_SHA256':c['parent_library_SHA256'],'new_bytes_per_owner':335544320,'default_OFF':True}
if __name__=='__main__':print(json.dumps(check()))
