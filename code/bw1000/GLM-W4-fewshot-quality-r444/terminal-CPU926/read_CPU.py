"""Stdlib check of original CPU926 observations; does not rerun its cases."""
import argparse
import hashlib
import json
from pathlib import Path

OBS='5b56ec5e361c4d23306aadac6069483047d6a094904b6cb2f7dc937d93cc159d'
MF='32c6bec05fe616671e9962b86788c1556582ddc7c71f4bd662f72bb1c4712b22'
RESULT='80ca253ddd0624e52c590066099c2ec584dd207565182fefe4b2423829bf1369'
sha=lambda b:hashlib.sha256(b).hexdigest()


def audit(observation,repo):
    raw=Path(observation).read_bytes();assert sha(raw)==OBS;o=json.loads(raw)
    assert o['accounting']['rc']==0 and o['accounting']['stdout']=='910407.926|GLM-r444-eight-slot-nodes-CPU-v1|COMPLETED|0:0|00:00:03|a14r2n20'
    entry=next(m for p,m in o['files'].items()if p.endswith('/CPU-result.json'))
    raw=entry['rawtext'].encode();assert len(raw)==90024 and sha(raw)==entry['SHA256']==RESULT;q=json.loads(raw)
    assert q['completed']and q['passed']and not q['errors']and q['actual_CPU']=='910407.926'
    assert q['SLURM_CPUS_PER_TASK']=='32'and len(q['actual_entry_mask'])==len(set(q['actual_entry_mask']))==32
    assert q['genuine_at_least32CPU_entry']and q['original_environment_restored']and q['entry_mask_unchanged']
    assert not q['GPU_node_qualification']and q['necessary_create_args_recording_only']
    assert not q['Native_dlopen']and not q['SDK_executed']and not q['weights_loaded']and not q['compile']
    assert not q['forbidden_modules_present']and not q['forbidden_maps']
    root=Path(repo)/'code/bw1000/GLM-W4-fewshot-quality-r444'
    raw=(root/'nodes-CPU-v1/SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==MF==q['source_manifest_SHA256'];m=json.loads(raw)
    for name,h in m['files_SHA256'].items():assert sha((root/name).read_bytes())==h,name
    slots=json.loads((root/'deployment-v1/SLOTS.json').read_bytes())['slots']
    assert len(q['cases'])==16 and {(c['slot'],c['stage'])for c in q['cases']}=={(s,t)for s in range(8)for t in range(2)}
    for c in q['cases']:
        s,t=c['slot'],c['stage'];env=c['Bash_then_actual_configure_env'];layout=c['layout']
        assert c['passed']and c['Bash_returncode']==0 and c['CAPI_recording_calls']==1
        assert c['recorded_hostname_leaf']==slots[s]['nodes'][t]==layout['node']
        assert json.loads(env['GLM_EXPECTED_NODES_R444'])==slots[s]['nodes']
        assert env['GLM_PP_SIZE_R3603']=='2'and env['GLM_CACHE_ROWS_R3627']=='1'and env['GLM_CACHE_CAP_R3627']=='32768'
        assert env['GLM_V_ON_DEMAND_R403']==env['GLM_BOUNDED_ROPE_R4279']==env['GLM_SHARED_CONTROL_R4279']=='1'
        assert env['SLURM_JOB_ID']+'.'+env['SLURM_STEP_ID']==q['actual_CPU']and env['SLURM_PROCID']==str(t)
        assert (layout['layer_begin'],layout['layer_end'])==((0,38)if t==0 else(38,78))
        assert c['original_eight_argument_create_contract']==dict(rank=0,stage=t,rows=1,capacity=32768,argument_count=8)
        assert c['actual_entry_mask_after']==q['actual_entry_mask']
    assert q['negative_contracts']==['wrong hostname','foreign/duplicate config nodes']
    ids=[]
    for p in q['preflight']:
        assert p['questions']==32 and p['HTTP']==64 and p['balanced_pair_order']
        assert len(p['actual_public_CLI_two_roles'])==2 and all(v['source_and_schema_valid']and not v['GPU']and not v['Native_dlopen']for v in p['actual_public_CLI_two_roles'])
        ids+=p['planned_case_ids']
    assert len(ids)==len(set(ids))==256 and sum(p['control_exposed']for p in q['preflight'])==160
    assert not q['CPU922_HTTP_and_CPU918_encoding_repeated']
    return dict(passed=True,actual_CPU='910407.926 COMPLETED0/3s',genuine_actual_entry_cpus=32,source_manifest_SHA256=MF,CPU_result_SHA256=RESULT,
        source_members=len(m['files_SHA256']),Bash_configure_layout_roles=16,original8arg_CAPI_recorded_calls=16,negative_contracts=2,
        source_CLI_roles=16,fixed_disjoint_questions=256,planned_HTTP=512,planned_exposed=160,unexposed_MMLU_kept=96,
        actual_Native=False,SDK_executed=False,GPU=False,Model=False,actual_GPU_node_qualification=False,
        scope='Actual CPU affinity/Bash/configure/purelayout, recorded hostname and create callable only. CPU926 and old deployment-v1 source are immutable; HOST v1 namespace failure is separately preserved and repaired by v2.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    q=audit(a.observation,a.repo);Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
