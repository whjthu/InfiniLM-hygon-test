"""Independent later scoring: never feeds labels to native/model preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
from contracts import canonical,need,score,strict_json


def generation_gate(rows):
    need(len({r['id'] for r in rows})==len(rows),'Duplicate output IDs')
    eligible={};reasons={};counts={}
    for row in rows:
        reason=None
        if row.get('status')!='COMPLETED':reason='not_completed'
        elif type(row.get('natural_EOS')) is not bool or type(row.get('cap_hit')) is not bool:reason='missing_or_untyped_termination_flags'
        elif row['cap_hit']:reason='cap_hit'
        elif not row['natural_EOS']:reason='no_explicit_natural_EOS'
        elif not isinstance(row.get('response'),str):reason='nontext_response'
        eligible[row['id']]=row.get('response') if reason is None else None
        if reason is not None:
            reasons[row['id']]=reason;counts[reason]=counts.get(reason,0)+1
    return eligible,dict(supplied_rows=len(rows),eligible_rows=len(rows)-len(reasons),unqualified_rows=len(reasons),
        rejection_counts=counts,rejections_by_id=reasons,typed_natural_EOS_required=True,cap_hit_must_be_false=True)


def main(labels,labels_sha,outputs,outputs_sha,output,split,blind_root_go):
    need(split in ('train','dev','blind') and (split!='blind' or blind_root_go),'Blind scoring not authorized by default')
    need(not output.exists(),'Scoring result already exists')
    need(labels.stat().st_size<=2097152 and outputs.stat().st_size<=4194304,'Scoring payload cap exceeded before read')
    labelraw=labels.read_bytes();outraw=outputs.read_bytes()
    need(hashlib.sha256(labelraw).hexdigest()==labels_sha and hashlib.sha256(outraw).hexdigest()==outputs_sha,'Frozen scoring identities differ')
    gold=[strict_json(line) for line in labelraw.splitlines()];rows=[strict_json(line) for line in outraw.splitlines()]
    eligible,generation=generation_gate(rows)
    # Missing/rejected/capped/non-natural/untyped responses cannot score; full216 remains.
    result=score(gold,eligible,216)
    result.update(split=split,labels_SHA256=labels_sha,outputs_SHA256=outputs_sha,formal17_result=False,
        exact_match_only=True,generation_qualification=generation,
        paired_effect='PENDING_FIXED_BASE_GENERIC_DOMAIN_ARMS',semantic_readability='UNKNOWN')
    output.write_bytes(canonical(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('labels','outputs','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('labels-sha','outputs-sha'):p.add_argument('--'+n,required=True)
    p.add_argument('--split',choices=['train','dev','blind'],default='dev');p.add_argument('--blind-root-go',action='store_true')
    a=p.parse_args();main(a.labels,a.labels_sha,a.outputs,a.outputs_sha,a.output,a.split,a.blind_root_go)
