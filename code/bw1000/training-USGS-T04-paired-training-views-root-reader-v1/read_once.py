"""Offline source+summary audit; R checker did actual rows, no PC caption body reads."""
from collections import Counter
import ast
import json
from pathlib import Path
from checks import canonical,check_producer,expected_indices,need,sha

G=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parent
SOURCE=G/'code/bw1000/training-USGS-T04-paired-training-views-CPU-v1'
OUT=G/'local-only/training-USGS-T04-paired-training-views-root-audit-v1'


def raw_pinned(pin):
    p=G/pin['path'];need(p.is_file() and not p.is_symlink(),'Pinned local original required');a=p.stat();raw=p.read_bytes();b=p.stat()
    need(a.st_size==b.st_size==pin['bytes'] and a.st_mtime_ns==b.st_mtime_ns and sha(raw)==pin['SHA256'],'Pinned original changed');return raw

def consume():
    binding=json.loads((S/'BINDING.json').read_bytes());need(binding['root_authorized_execute_now'] is True and isinstance(binding['independent_original'],dict),'Wait for actual root independent package')
    receipt=json.loads(raw_pinned(binding['original']['receipt.json']));prod=json.loads(raw_pinned(binding['original']['stdout.private']));check_producer(receipt,prod,binding)
    independent=json.loads(raw_pinned(binding['independent_original']));rr=json.loads(raw_pinned(binding['independent_receipt']));remote_source=raw_pinned(binding['independent_source'])
    need(rr['passed'] is True and rr['rc']==0 and rr['stdout_SHA256']==binding['independent_original']['SHA256'] and rr['stderr_bytes']==0,'Independent R original not successful')
    need(sha(remote_source)=='12f26bb6f50ab765c4a5d3281e561670385dd88c0a026fc8bd4ddfad35c6e1d7','Reviewed R checker source differs');ast.parse(remote_source)
    need(independent['passed'] is True and independent['request_id']==binding['request_id'] and independent['original_epoch']==binding['epoch']
        and independent['original_owner_exited_same_host_boot'] is True and independent['original_result']==prod,'Independent original process/context differs')
    need(independent['new_encodes']==independent['model_calls']==independent['GPU_calls']==0 and independent['raw_row_body_exported'] is False
        and len(independent['CPU_affinity'])==2 and not set(independent['CPU_affinity'])&{0,1},'Independent checker scope differs')
    files=independent['files'];need(len(files)==17 and sum(p['bytes'] for p in files.values())==independent['logical_read_bytes']==2559940 and independent['logical_read_bytes']<=8*1024**2,'Actual R bounded read scope differs')
    rootstr=json.loads((SOURCE/'PLAN.json').read_bytes())['namespace']
    def remote_pin(path,pin):
        got=files[path];need(got['bytes']==pin['bytes'] and got['SHA256']==pin['SHA256'],'Independent actual R artifact identity differs')
    remote_pin(rootstr+'/paired-result.json',dict(bytes=len(canonical(prod)),SHA256=sha(canonical(prod))))
    mf=(SOURCE/'SOURCE_FILES.json').read_bytes();need(sha(mf)==prod['source_manifest_SHA256'],'Actual source MF changed')
    for name,pin in json.loads(mf)['files'].items():
        raw=(SOURCE/name).read_bytes();need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'Source member differs')
    need(sha((SOURCE/'CLAIM.json').read_bytes())==prod['claim_SHA256'],'Claim differs')
    plan_raw=(SOURCE/'PLAN.json').read_bytes();plan=json.loads(plan_raw);remote_pin(rootstr+'/src/PLAN.json',dict(bytes=len(plan_raw),SHA256=sha(plan_raw)))
    indices=expected_indices();counts={str(k):v for k,v in sorted(Counter(indices).items())};seqsha=sha(canonical(indices));arms={}
    for storage,observer in [('domain','domain'),('control','generic')]:
        a=prod['arms'][storage];r=independent['arms'][storage];spec=plan['arms'][storage];base=rootstr+'/'+storage
        need(r['complete_original_rows']==45 and r['complete_derived_rows']==r['exact_source_line_copies']==800 and r['counts']==counts
            and r['sequence_SHA256']==seqsha and r['CPU_target_count_available']==(storage=='control'),'Offline expected schedule/frequencies differ')
        remote_pin(spec['canonical45']['path'],spec['canonical45']);remote_pin(spec['native_identities']['path'],spec['native_identities'])
        need(a['source45_SHA256']==spec['canonical45']['SHA256'] and a['native_reference_SHA256']==spec['native_identities']['SHA256'],'Qualified source link differs')
        for kind in ['view','occurrences','occurrence_ledger']:
            pin=a[kind];remote_pin(base+'/'+pin['name'],pin)
        primary=canonical(dict(arm=observer,view_SHA256=a['view']['SHA256'],original_index_by_physical_row=indices))
        need(len(primary)==a['occurrences']['bytes'] and sha(primary)==a['occurrences']['SHA256'],'Offline reconstructed exact observer dict differs')
        remote_pin(base+'/source45-identities-private.json',dict(bytes=files[base+'/source45-identities-private.json']['bytes'],SHA256=a['source_reference45_SHA256']))
        remote_pin(base+'/view-binding-private.json',dict(bytes=len(canonical(a)),SHA256=sha(canonical(a))))
        need(r['view']==files[base+'/train-800.jsonl'] and r['occurrences']==files[base+'/exposure-ledger-private.json'],'Independent per-arm actual files metadata differs')
        arms[storage]=dict(observer_arm=observer,qualified_unique45_source=spec['canonical45'],source_native_reference=spec['native_identities'],view=a['view'],occurrences=a['occurrences'],
            row_occurrence_witness=a['occurrence_ledger'],source45_reference_SHA256=a['source_reference45_SHA256'],actual_complete_original_rows=45,actual_source_line_copies_R_verified=800,
            fixed_sequence_SHA256=seqsha,minimum_exposures_per_original=17,maximum_exposures_per_original=18,counts_45_SHA256=sha(canonical(counts)),
            CPU_reference_input_tokens_unique45=a['CPU_reference_input_tokens_unique45'],CPU_reference_input_tokens_weighted800=a['CPU_reference_input_tokens_weighted800'],
            CPU_reference_effective_supervised_tokens_unique45=a['CPU_reference_effective_supervised_tokens_unique45'],CPU_reference_effective_supervised_tokens_weighted800=a['CPU_reference_effective_supervised_tokens_weighted800'],
            actual_GPU_rank_mapping_padding_cursor_qualified=False,actual_training_updates=0,CPU_reference_identity_candidates_only=True)
    OUT.mkdir();result=dict(passed=True,state='PAIRED800_DATA_VIEWS_INDEPENDENT_METADATA_SOURCE_SUMMARY_PASS_NO_TRAINING',request_id=binding['request_id'],source_publication=binding['publication'],epoch=binding['epoch'],
        original_process_rc=receipt['rc'],original_owner_exited_same_host_boot=True,planned_arms=2,qualified_unique_rows_per_arm=45,actual_unique_source_rows=90,
        derived_rows_per_arm=800,actual_new_view_rows=1600,actual_training_updates=0,new_encodes=0,model_calls=0,GPU_cards=0,network_downloads=0,
        arms=arms,observer_occurrences_schema=['arm','view_SHA256','original_index_by_physical_row'],control_storage_directory_observer_arm='generic',
        independent_R_verified='All1600 view lines equal original45 lines[index]; all1600 occurrence metadata physical/index/sourceID/nativeRecordSHA/input_ids/labels references; same schedule/17or18; original owner exited same host/boot.',
        offline_verified='Reviewed exact R checker source and actual returned metadata; locally reconstructed800-index sequence,45counts and both primary dict ledger byteSHA. Did not receive or reread caption/view bodies or full1600 metadata arrays on PC.',
        remaining_candidate_field_boundary='R checker directly compares collated input_ids/labels. Other encoded/attention/vision candidate fields are source-bound metadata, not separately GPU-validated. No actual training rank cursor/padding equality inferred.',
        raw_caption_or_train800_body_exported_to_PC=False,full800_rows_independently_reread_on_PC=False,dev_blind_input_or_gold_read=False,
        domain_effective_token_counter=None,domain_token_count_instrumentation_missing_preserved=True,control_unique45_actual_CPU_reference_tokens=694,
        token_matching_between_arms_qualified=False,actual_training_token_exposures='NOT_RUN',
        CPU_affinity_original=prod['CPU_affinity_actual'],CPU_affinity_independent_reader=independent['CPU_affinity'],original_self_device_FDs=prod['self_device_FDs'],original_spawned_children=0,
        original_monotonic_client_elapsed_seconds=receipt['elapsed_seconds'],independent_reader_monotonic_elapsed_seconds=independent['elapsed_seconds'],
        original_finished_reported_remote_wall=prod['finished'],duration_from_cross_host_wall_subtraction_claimed=False,original_peak_self_RSS_bytes=prod['peak_self_RSS_bytes'],RSS_hard_limit=False,
        original_epoch_budget_seconds=120,independent_R_metadata_logical_read_bytes=2559940,independent_R_metadata_file_count=17,
        source_manifest_SHA256=sha(mf),claim_SHA256=prod['claim_SHA256'],original_invocation=binding['original'],independent_package=binding['independent_original'],independent_source=binding['independent_source'],independent_request_receipt=binding['independent_receipt'],
        reader_SHA256=sha(Path(__file__).read_bytes()),checks_SHA256=sha((S/'checks.py').read_bytes()),native_or_exposure_fixture_tests_rerun=0,
        Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False,formal17_performance=False,domain_capability_evaluated=False)
    (OUT/'result.json').write_bytes(canonical(result));print(json.dumps({k:result[k] for k in ['passed','actual_unique_source_rows','actual_new_view_rows','actual_training_updates','new_encodes','domain_token_count_instrumentation_missing_preserved']},sort_keys=True));return result

if __name__=='__main__':consume()
