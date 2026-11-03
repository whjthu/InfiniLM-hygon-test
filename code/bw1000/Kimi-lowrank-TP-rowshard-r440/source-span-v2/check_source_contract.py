"""Actual new production source consumer; no project import, SDK or devices."""
from pathlib import Path
import hashlib,json,re
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 refs=json.loads((D/'SOURCE_REFERENCES.json').read_bytes());oldroot=Path('/refactor/incoming/Kimi-full-MoE-lowrank-M1-r437')
 for p,h in refs['parent_source_files'].items():assert sha(p)==h,p
 old=(oldroot/'private_moe.cpp').read_text().replace('KimiK3MoER437','KimiK3MoER440').replace('_private_r437','_private_r440')
 expected=re.sub(r'kimi_r437::project\(runtime, (routed_expert_(?:down|up)_proj_), (flattened|input_row|row|routed), ([01])\)',r'kimi_r440::project_scoped(runtime, \1, \2, \3,tp_size_,communicator_)',old)
 actual=(D/'private_moe.cpp').read_text();assert actual==expected and actual.count('kimi_r440::project_scoped')==5
 assert (D/'private_moe.hpp').read_text()==(oldroot/'private_moe.hpp').read_text().replace('KimiK3MoER437','KimiK3MoER440')
 assert (D/'owner_affinity.hpp').read_text()==(oldroot/'cpu-owner-v4/owner_affinity.hpp').read_text().replace('r437','r440')
 # Actual TP allreduce + norm remain between the different down/up inputs.
 assert actual.index('mark("routed_allreduce")')<actual.index('routed = routed_expert_norm_->forward')<actual.index('kimi_r440::project_scoped(runtime, routed_expert_up_proj_')
 assert 'FusedMoeMxfp4' in actual and 'shared_experts_->forward' in actual
 p=(D/'projection.hpp').read_text();assert p.index('scope->require')<p.index('dense_projection(')<p.index('scope->retain(t)')<p.index('allgather_(')
 assert 'input->contiguous(r)' in p and 'shard->contiguous(r)' in p and 'weight->narrow' in p
 c=(D/'component.cpp').read_text();a=c.index('const auto begin=Clock::now();rt::Tensor y;');b=c.index('const double native_ms=ms(begin);',a);timed=c[a:b]
 assert timed.count('ProjectionLeaseScope')==1 and timed.index('candidate->forward')<timed.index('scope.complete_original_forward()')
 assert 'r.syncStream()'not in timed and 'retained_handles=scope.retained_before_complete();}'in timed
 assert 'entry.size()<32'in c and c.index('Runtime::create')<a<c.index('actual original firstsubmit owner subset mismatch')
 scope=(D/'lease_scope.hpp').read_text();assert '[&r]{r.syncStream();}'in scope and 'active_scope!=this'in scope and 'comm!=comm_'in scope and 'rank!=node*8+local'in scope
 pending=(D/'pending_leases.hpp').read_text();assert 'drain_();refs_.clear();std::vector<T>().swap(refs_);closed_=true;'in pending
 assert 'kimi_r437'not in actual+c+p and 'hits'not in p and 'gemv'not in p.lower()
 return dict(passed=True,private_class_field_layout_unchanged=True,whole_original_MoE_body_except_exact_five_projection_calls=True,shared_router_expert_norm_TP_math_preserved=True,exact_same_owner_first_submit_policy=True,caller_owns_leases_through_original_sync=True,release_and_scope_destruction_in_original_primary_interval=True,changed_CPU_only=True,SDK_execution=False,new_numeric_qualification=False)
if __name__=='__main__':print(json.dumps(main()))
