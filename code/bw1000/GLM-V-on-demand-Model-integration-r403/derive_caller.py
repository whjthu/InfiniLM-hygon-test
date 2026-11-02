"""Exact source derivation from actual038d; parent/header/layout never edited."""
from pathlib import Path
import hashlib,json
from derive_parent_r398 import derive as previous
D=Path(__file__).resolve().parent
def derive(raw):
 baseline,old=previous(raw);s=baseline.decode();changes=[]
 def edit(a,b):
  nonlocal s
  assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b);changes.append(dict(old=a,new=b))
 edit('#include "selected_values_r398.hpp"','#include "policy_r403.hpp"\n#include "host_ids_r403.hpp"\n#include "selected_values_r400.hpp"')
 edit('v_on_demand_r398::requested()','v_on_demand_r403::requested()')
 assert s.count('v_on_demand_r398::selected_values')==2
 s=s.replace('v_on_demand_r398::selected_values','v_on_demand_r400::selected_values')
 changes.append(dict(change='Both original selected-V callsites use unchanged tested r400 helper; fixed8192'))
 # C1/hostbankOFF makes the actual batch5055 predicate false even when the
 # inherited GLM_BATCH_ALLSELECTED_R5096=1 flag is present. Preserve that flag.
 edit('v398&&(resident5640||request_batch5055||host_bank_r4504_||rows_!=1||capacity_!=32768)',
      'v398&&(resident5640||host_bank_r4504_||rows_!=1||capacity_!=32768)')
 edit('struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;',
      'struct Work{rt::Runtime&r;v_on_demand_r403::HeldHostIds host_ids;std::vector<rt::Tensor>held;')
 edit('rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}',
      'rt::Tensor ids(const std::vector<int64_t>&v){const auto&owned=host_ids.hold(v);auto x=alloc({owned.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),owned.data(),owned.size()*8,false);return x;}')
 # Preserve all other Model/indexer/attention/state statements, including OFF.
 assert s.count('r.syncStream()')==baseline.decode().count('r.syncStream()')
 return s.encode(),dict(parent_edits=old,integration_edits=changes,HOST_TUs_changed=1,headers_class_layout_changed=False,added_sync=0,default_OFF=True)
if __name__=='__main__':
 p=D.parent/'agent-context-32k/GLM-PP4-registry-repair-private-r184b/resident_caller_registry_r184b.cpp'
 b,edits=derive(p.read_bytes());(D/'caller_v_on_demand_r403.cpp').write_bytes(b)
 (D/'DERIVATION.json').write_text(json.dumps(dict(parent_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),derived_SHA256=hashlib.sha256(b).hexdigest(),**edits),indent=2)+'\n')
