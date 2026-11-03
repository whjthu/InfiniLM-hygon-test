"""Stdlib rehash of CPU922 source/receipt/log; no HTTP or Model replay.

The original checker deleted its TemporaryDirectory. Recorded per-scenario
summaries/assertions are auditable; 97 individual wire/Run files are absent.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

ARCHIVE='042c046c894c5c982ac512f6ecdac8b5c68c515026fc2f088c99f61fd0f98787'
OBSERVATION='fbd0ff179fd2a69cebddb1638622a4f476c817cadf785a56b3dd0050ebf8936e'
MF='78faddcf6184a9adb2a66b8e8c38fa63444df1580b2247dba0fdc0f981bf2702'
PROVIDER='6947fb51ddb8462864ef12ae897d918595c36d3a0327cd2f118016d65c95473d'
RESULT='bf5a45d27f0cf0b664898282364c0925cf4b9c7c4d70454359b9831d45486c9c'
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
D=R+'/incoming/GLM-W4-fewshot-quality-r444-client-v1'
B=R+'/candidates/GLM-W4-fewshot-quality-r444-client-CPU-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()


def audit(archive,observation,repo):
    raw=Path(archive).read_bytes();assert sha(raw)==ARCHIVE
    a=json.loads(gzip.decompress(raw));assert len(a['files'])==141 and a['new_Slurm']==a['GPU']==0 and not a['Native_dlopen']
    assert not a['ephemeral_recording_HTTP_and_Run_fixture_files_preserved']
    files={}
    for name,m in a['files'].items():
        data=base64.b64decode(m['base64'],validate=True);assert len(data)==m['bytes']and sha(data)==m['SHA256'];files[name]=data
    raw=files[D+'/client-v1/SOURCE_MANIFEST.json'];assert sha(raw)==MF;m=json.loads(raw)
    assert len(m['files_SHA256'])==136 and not m['GPU_GO']and not m['Model_GO']
    for name,h in m['files_SHA256'].items():assert sha(files[D+'/'+name])==h
    p=json.loads(files[D+'/execution-v1/SOURCE_MANIFEST.json']);assert sha(files[D+'/execution-v1/SOURCE_MANIFEST.json'])==PROVIDER
    assert len(p['files_SHA256'])==127
    repo=Path(repo);parent_raw=(repo/'code/bw1000/public-GLM-native-provider-r425/SOURCE_MANIFEST.json').read_bytes()
    assert sha(parent_raw)==p['parent_source_manifest_SHA256']=='f5015c22a4434576a9b4cf00708dbe19594b70598a6b3e004410bfc63dbd9858'
    parent=json.loads(parent_raw)
    changes=[]
    for name,h in parent['files_SHA256'].items():
        assert sha((repo/'code/bw1000/public-GLM-native-provider-r425'/name).read_bytes())==h
        if sha(files[D+'/execution-v1/'+name])!=h:changes.append(name)
    assert set(changes)=={'config.template.json','public_glm_native/cli.py','public_glm_native/native_owner.py'}
    before=(repo/'code/bw1000/public-GLM-native-provider-r425/public_glm_native/cli.py').read_text()
    after=files[D+'/execution-v1/public_glm_native/cli.py'].decode().replace('\r\n','\n')
    assert before.replace('== (5100, 4500)','== (28800, 28200)')==after
    original_mf=json.loads((repo/'code/bw1000/GLM-W4-fewshot-quality-r444/SOURCE_MANIFEST.json').read_bytes())
    for new,old in [('public_glm_native/native_owner.py','native_owner_audited.py'),('public_glm_native/prefill_window_audit.py','prefill_window_audit.py')]:
        assert sha(files[D+'/execution-v1/'+new])==original_mf['files_SHA256'][old]
    q_raw=files[B+'/whole-main-result.json'];assert sha(q_raw)==RESULT;q=json.loads(q_raw)
    assert q['passed']and q['completed']and not q['errors']and q['actual_HTTP']==97
    assert q['original_encoding_SHA256']=='1e083b8241422ed87511d83fa8585bfc19eb356e539d42acfa80b4c10cc8351c'
    expected={'positive':(88,88,0),'foreign_ready':(0,0,1),'bad_counter':(1,0,1),'same_run_STOP':(4,4,1),'HTTP_error':(4,3,1)}
    assert len(q['phases'])==len(expected)
    results=[]
    for phase in q['phases']:
        assert phase['passed']and phase['name']in expected
        assert(phase['actual_HTTP'],phase['rows'],phase['client_returncode'])==expected[phase['name']]
        pooled=phase['result_summary']['pooled'];assert pooled['expected_pairs']==44
        assert sum(v['planned']for v in pooled['arms'].values())==88
        assert sum(v['observed']for v in pooled['arms'].values())==phase['rows']
        assert sum(v['uncompleted_or_not_attempted']for v in pooled['arms'].values())==88-phase['rows']
        assert pooled['planned_denominator_never_shrinks']and not phase['result_summary']['noninferiority_proven']
        if phase['name']=='positive':
            assert pooled['observed_complete_pairs']['complete_pairs']==44
            assert sum(v['correct']for v in pooled['arms'].values())==85
            assert sum(v['raw_answer_matches_gold']for v in pooled['arms'].values())==86
            assert sum(v['caps']for v in pooled['arms'].values())==1 and sum(v['parse_failures']for v in pooled['arms'].values())==1
        results.append(dict(name=phase['name'],HTTP=phase['actual_HTTP'],rows=phase['rows'],planned_HTTP=88,passed=True))
    assert all(q[k]for k in ['actual_whole_client_main','identity_not_mocked','actual_Run_and_all16_ready_consumed','actual_original_executor_observed_with_recording_Native_leaves','real_CPU918_encodings_reused_no_encode','quality_wrong_cap_parse_continue','partial_planned_denominators_kept','same_run_STOP_no_next_request'])
    assert not q['qualified_live_Model_or_W4_counter']and not q['old64_suite_replayed']
    assert q['actual_GPU_hiding_env']=={k:'-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}
    assert not any(q[k+'_imported']for k in ['torch','transformers','_infinilm'])and not q['project_Native_maps']
    assert len(q['actual_private_entry_CLI_both_roles'])==2 and all(r['source_and_schema_valid']and not r['Native_dlopen']and not r['GPU']for r in q['actual_private_entry_CLI_both_roles'])
    obsraw=Path(observation).read_bytes();assert sha(obsraw)==OBSERVATION;obs=json.loads(obsraw)
    assert obs['accounting']['rc']==0 and obs['accounting']['stdout']=='910407.922|GLM-r444-whole-client-CPU-v1|COMPLETED|0:0|00:00:43|a14r2n20'
    assert obs['files'][B+'/whole-main-result.json']['rawtext'].encode()==q_raw
    intent=json.loads(files[R+'/results/GLM-r444-whole-client-CPU-v1-intent.json']);assert intent['attempts']==1 and intent['source_manifest_SHA256']==MF and intent['publication_commit']=='99a3fbea6d88a773331c47c4401022605559268e'
    return dict(passed=True,actual_CPU='910407.922 COMPLETED0/43s',archive_SHA256=ARCHIVE,original_files=141,source_members=136,private_provider_members=127,
        source_manifest_SHA256=MF,provider_manifest_SHA256=PROVIDER,CPU_result_SHA256=RESULT,actual_HTTP_recording=97,scenarios=results,
        actual_main_Run_ready_owner_counter_and_private_CLI_contract=True,real_CPU918_encoding_reused=True,
        scope='Pure CPU contract with real localhost HTTP and actual Run/executor/identity/source; Native/loaded owners/model response leaves explicitly recorded.',
        original_deleted_recording_wire_files_available=False,limitation=a['ephemeral_limit'],
        Native_dlopen=False,GPU=False,Model=False,actual_service_qualification=False,live_W4_exposure_qualified=False,quality_result=False,GPU_GO=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--observation',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    q=audit(a.archive,a.observation,a.repo);Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))


if __name__=='__main__':main()
