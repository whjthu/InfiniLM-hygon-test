"""Pure source/integer bounds; actual class behavior is tested by the C++ fixture."""
import pathlib,json,hashlib
from long_contract_r177 import geometry,contiguous_prefill
D=pathlib.Path(__file__).resolve().parent
def check():
 c=json.loads((D/'source-context-r180.json').read_bytes());new=(D/'include/native_sparse_cache_r22.hpp').read_bytes()
 assert hashlib.sha256(new).hexdigest()==c['new_header_SHA256']
 old=new.replace(b'capacity>65536',b'capacity>4096').replace(b'capacity1..65536',b'capacity1..4096')
 assert hashlib.sha256(old).hexdigest()==c['original_header_current_SHA256']
 assert new.count(b'capacity>65536')==1 and b'rows>128'in new
 assert c['per_forward_T_cap']==4096 and c['prefill_chunk']==1024
 bounds=[]
 for capacity in [32768,65536]:
  for rows in [1,2]:
   g=geometry(capacity,rows);assert g['source_page_limits_admissible']
   h=(capacity+127)//128;cs=(capacity+3)//4
   # Host wrappers: latent gather max65536, index gather max262144 and
   # per-op index chunk65536, status writes/slots65536. No integer wrap.
   assert rows*h<=65536 and rows*cs<=65536 and rows*cs<=262144
   assert rows*min(capacity,127)<=65536
   assert c['prefill_chunk']<=4096 and (c['prefill_chunk']+3)//4<=65536
   assert capacity+8192<2**31 and rows*cs<2**63
   plan=contiguous_prefill(capacity-8192,1024);assert plan[-1]['total']==capacity-8192
   bounds.append(dict(capacity=capacity,rows=rows,geometry=g,C128_history_slots_at_capacity=rows*h,C4_and_index_history_slots_at_capacity=rows*cs,per_forward_T_cap_unchanged=4096,real_prefill_chunk=1024,integer_boundaries_pass=True))
 return dict(passed=True,scope='SOURCE/INTEGER CONTRACT ONLY; no Native creation, Kernel execution, long-context task or quality qualification',guard_only_change=True,class_layout_changed=False,formal_rows=1,formal_context=32768,extension_context=65536,limits=bounds,source_vs_backtime='Current1198 headerSHA is source-confirmed; original ac937 compile-time headerSHA remains UNKNOWN. New build must recompile all fresh header consumers and remove every old inline guard from candidate.',new_Native_unqualified_until='Fresh complete HOST compile/MD/link closure; then separately authorized actual allocation/long-prefill/long-output/8tool/quality validation. No GPU GO in this package.')
if __name__=='__main__':print(json.dumps(check()))
