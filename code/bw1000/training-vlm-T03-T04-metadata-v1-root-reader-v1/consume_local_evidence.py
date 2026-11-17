"""Consume the original local transport and independent read-only reconciliation evidence."""
from pathlib import Path
import ast
import hashlib
import json

G=Path(__file__).resolve().parents[3]
REQUEST='bc17d9a5727f48f58240c5bdd1af9d39'
PUB='3155d5b88cbafdad6f349a8743c9ef6d3c08536c'
MF='cda6393b110183f25f83ae20e3b91669988535a6cf9e1665817f11af7dbf2166'
H=lambda b:hashlib.sha256(b).hexdigest()
PINS={}
def read(relative):
    p=G/relative
    b=p.read_bytes()
    assert len(b)<=262144
    PINS[relative]=dict(bytes=len(b),SHA256=H(b))
    return json.loads(b)
def main():
    rel='local-only/training-vlm-T03-T04-metadata-v1/'
    intent=read(rel+'intent.json');collector=read(rel+'outcome.json')
    source=read(rel+'source-transport.json');receiver=read(rel+'receiver-transport.json')
    sr=read('local-only/training-vlm-T03-T04-metadata-v1-source-terminal-first/receipt.json')
    tr=read('local-only/training-vlm-T03-T04-metadata-v1-receiver-reconcile-first/receipt.json')
    assert intent['request_id']==collector['request_id']==REQUEST
    assert intent['publication']==PUB and intent['source_manifest_SHA256']==MF
    assert intent['deadline']==intent['epoch']+300 and collector['passed'] is False
    assert source['returncode']==0 and source['remote_outcome']=='RETURNED' and source['stdout_bytes']==7937372
    assert receiver['returncode']==1 and receiver['remote_outcome']=='UNKNOWN' and not receiver['timed_out']
    assert receiver['stream_errors']==['BrokenPipeError'] and receiver['stdout_bytes']==0
    stderr=(G/rel/'receiver-stderr.private').read_bytes()
    assert len(stderr)==373 and H(stderr)==receiver['stderr_SHA256']=='86687201f1aefe0a876c1b2ba34e05086eafb05e98d599ab285cdf760ff4c35f'
    assert b'published-metadata-endpoint' in stderr and b'in run_receiver' in stderr and b'in parent_reservations' in stderr
    assert b'ValueError: parent no longer owned RUNNING40' in stderr
    # This is a real remote Python failure, not an inferred nonexecution from an absent receipt.
    D=G/'code/bw1000/training-vlm-T03-T04-metadata-v1'
    raw=(D/'SOURCE_FILES.json').read_bytes();assert H(raw)==MF
    for name,pin in json.loads(raw)['files'].items():
        b=(D/name).read_bytes();assert len(b)==pin['bytes'] and H(b)==pin['SHA256']
    code=(D/'receive.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='run_receiver')
    body=ast.get_source_segment(code,node)
    assert body.index('reservation = parent_reservations(plan)')<body.index('identity = own(root')
    assert body.index('identity = own(root')<body.index('sys.stdin.buffer.read')
    assert sr['rc']==tr['rc']==0
    sv,tv=sr['value'],tr['value'];s=sv['source']
    assert s['request_id']==REQUEST and s['publication']==PUB and s['source_manifest_SHA256']==MF
    assert s['passed'] is True and s['source_bytes_read']==32327445 and s['source_files_read']==67 and s['member_count']==36
    assert s['bundle_SHA256']==source['stdout_SHA256'] and s['bundle_bytes']==source['stdout_bytes']
    assert s['weight_payload_bytes_read']==s['initial_tensor_bytes_read']==s['actual_GPU_cards']==0
    assert sv['source_process']['original_owned_process_live'] is False
    assert tv['observed_epoch']>intent['deadline'] and tv['receiver_namespace_exists'] is False and tv['receiver_records']=={}
    assert len(tv['parents'])==2
    assert all(p['queue_state']==p['scontrol_state']=='RUNNING' and p['queue_nodes']==p['scontrol_nodes']=='40'
               and p['owner_uid_matches'] is True and 'gres/dcu=320' in p['AllocTRES'] for p in tv['parents'])
    result=dict(state='ORIGINAL_METADATA_SOURCE_PASS_RECEIVER_TERMINAL_PRE_NAMESPACE_FAIL_RECONCILED',
       request=REQUEST,source_publication=PUB,source_manifest_SHA256=MF,epoch=intent['epoch'],deadline=intent['deadline'],
       root_elapsed_seconds=collector['finished_epoch']-intent['epoch'],
       original_controller_passed=False,original_transport_receiver_state='UNKNOWN',
       independent_receiver_state='REMOTE_PYTHON_EXIT1_PARENT_GATE_BEFORE_NAMESPACE_AND_BUNDLE_READ',
       receiver_error='ValueError: parent no longer owned RUNNING40',
       remote_exit_code=1,stderr_SHA256=receiver['stderr_SHA256'],
       source_passed=True,source=s,source_original_record_pins=sv['original_records'],
       source_owned_process_absent=True,source_post=sv['source_process'],
       root_compressed_bytes_received=source['stdout_bytes'],root_compressed_SHA256=source['stdout_SHA256'],
       source_bundle_persisted=False,source_derived_initial_bindings_persisted=False,
       receiver_namespace_absent_after_original_deadline=True,
       zero_target_metadata_file_writes_basis='Remote Python trace exits inside parent_reservations, before own()/payload read, combined with exact frozen source and postdeadline absent namespace; absence alone is not nonexecution proof.',
       target_written_members=0,target_member_denominator=36,target_initial_rows_received=0,initial_row_denominator=16,
       source_named_initial_values_not_available_to_target=True,planned_source_logical_read_bytes=32327445,
       application_consumed_weight_payload_bytes=0,application_consumed_initial_tensor_bytes=0,
       BufferedReader_and_OS_readahead_physical_IO='UNKNOWN_NOT_MEASURED',
       current_parent_observation=tv,current_parent_reserved_DCU_sum=640,
       current_unique80_nodes_not_rechecked_by_this_reconciliation=True,
       reported_NodeCount_sum=80,no_new_Slurm_or_GPU=True,new_GPU_steps=0,Torch_Native_SDK_calls=0,
       full_weight_identity=False,actual_target_CPU_model_qualification=False,formal_performance=False,formal_quality=False,
       original_v1_closed_no_replay=True,original_transport_and_failure_denominators_retained=True,evidence=PINS)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
