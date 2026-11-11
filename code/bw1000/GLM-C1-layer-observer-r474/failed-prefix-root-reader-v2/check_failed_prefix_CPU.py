"""New failed-prefix boundaries. Fixtures have no real model inputs or device execution."""
import copy
from read_failed_prefix import completed_prefix, cross_prefix
import frozen_terminal as f

checks=[]
def reject(name,call):
    try:call()
    except (AssertionError,KeyError,ValueError):checks.append(name);return
    raise AssertionError('malformed evidence accepted: '+name)

done=dict(completed=True,reset_before=True,reset_after=True)
partial=dict(completed=False,reset_before=True,reset_after=False)
assert completed_prefix([done]*31+[partial])==31
assert completed_prefix([])==0 and completed_prefix([done]*33)==33
reject('completion_after_partial',lambda:completed_prefix([done,partial,done]))
reject('two_partial',lambda:completed_prefix([partial,partial]))
reject('completed_without_after_reset',lambda:completed_prefix([dict(done,reset_after=False)]))
reject('partial_with_after_reset',lambda:completed_prefix([dict(partial,reset_after=True)]))
reject('lost_before_reset',lambda:completed_prefix([dict(partial,reset_before=False)]))
reject('over33',lambda:completed_prefix([done]*34))

def fixture(n=31):
    rows=[dict(index=i,ordinal=i,suite='fixture',diagnostic=i==0,screen_index=None if i==0 else i-1,
               prompt_tokens=513,prefill_fragments=[512,1],generated_including_EOS=4,
               output_IDs_SHA256='a'*64,natural_EOS=True,cap_hit=False) for i in range(n)]
    ranks=[dict(rank=i,node=f.NODES[i//8],PCI='fixture'+str(i%8),identity_passed=True,
                validated=copy.deepcopy(rows)) for i in range(16)]
    phases=['created','loaded']+['case'+str(i) for i in range(n)]
    workers=[dict(node=f.NODES[i],actual_GPU=f.GPU,gpu_epoch=f.EPOCH,
                  rounds=[dict(phase=p,digest=str(j)) for j,p in enumerate(phases)]) for i in range(2)]
    return ranks,workers

r,w=fixture();assert cross_prefix(r,w)==31
r[15]['validated'].pop();assert cross_prefix(r,w)==30
for name,modify in [
    ('output_hash_disagreement',lambda r,w:r[8]['validated'][5].update(output_IDs_SHA256='b'*64)),
    ('token_count_disagreement',lambda r,w:r[8]['validated'][5].update(generated_including_EOS=5)),
    ('diagnostic_quality_mix',lambda r,w:r[8]['validated'][0].update(diagnostic=False)),
    ('rank_order',lambda r,w:r.reverse()),
    ('duplicate_PCI',lambda r,w:r[1].update(PCI=r[0]['PCI'])),
    ('bad_identity',lambda r,w:r[1].update(identity_passed=False)),
    ('missing_rank',lambda r,w:r.pop()),
    ('wrong_GPU',lambda r,w:w[0].update(actual_GPU='910371.900')),
    ('wrong_epoch',lambda r,w:w[0].update(gpu_epoch=0)),
    ('wrong_node',lambda r,w:w[0].update(node='fixture-foreign')),
    ('barrier_missing',lambda r,w:w[0]['rounds'].pop()),
    ('barrier_digest_mismatch',lambda r,w:w[0]['rounds'][3].update(digest='different')),
    ('barrier_order',lambda r,w:w[0]['rounds'].reverse()),
]:
    r,w=fixture();modify(r,w);reject(name,lambda:cross_prefix(r,w))
assert len(checks)==19
print('PASS: 19 failed-prefix/cohort malformed-evidence rejections; 31/33 and 30/32 remain partial, no model/GPU execution.')
