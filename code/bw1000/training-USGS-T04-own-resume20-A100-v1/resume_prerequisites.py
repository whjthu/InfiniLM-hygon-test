"""Small original-lineage/typedCPU/continuous100 admission; no model/state bodies."""
import hashlib
import json
from pathlib import Path


def need(ok,code):
    if not ok:raise ValueError(code)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def exact_json(pin,check):
    check();p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and 0<pin['bytes']<=2*1024**2,'PRIOR_JSON_BOUND')
    a=p.stat();raw=p.read_bytes();b=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'PRIOR_JSON_CHANGED');return json.loads(raw)

def config_contract(c,prior):
    need(c['case']=='T04.sft' and c['mode']=='resume20to25' and c['arm']=='domain' and c['pair_id']=='d58ebcf4dc0345b2a3719067f0fe6961','ONLY_OWN_ORIGINAL_DOMAIN_RESUME')
    need(c['world_size']==8 and c['physical_rows']==800 and c['unique_rows']==45 and c['accumulation']==4 and c['max_steps']==25,'UNCHANGED_DOMAIN_DENOMINATOR')
    for field,digest in [('native45','07f0073299aa3ba5f82b359affcc09704f9e8515205f7f4b87ae9c835884a65c'),('training800','57640ac8c8fd9a7d0614df276af37e0c92201ec293a3ea722fd0b6252ac01570'),('occurrences','1b15a612ec1beb014c11e7e2beb86e54aae6876c7e5cb17c9466758a47f62338')]:need(c[field]['SHA256']==digest,'ORIGINAL_DOMAIN800_ONLY_NO64')
    for step,key,rootkey in [('20','checkpoint_files','checkpoint'),('25','comparison_files','comparison_checkpoint')]:
        expected=prior['original_checkpoint20_25_pins'][step];need(c[key]==expected['files'] and c[rootkey]==expected['root'] and len(c[key])==14,'EXACT_OWN14_STATE_REFERENCE')
    need(c['initial_state']==prior['original_full_initial_pin'] and c['initial_state']['named_SHA256']=='086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137','OWN_PAIR_INITIAL_PIN')
    need(set(c['continuous_reference'])=={str(r) for r in range(8)} and Path(c['train_output']).name=='train','ALL8_OWN_CONTINUOUS_REFERENCE')
    need(c['checkpoint']!=c['train_output'] and c['comparison_checkpoint']!=c['train_output'] and Path(c['checkpoint']).parent!=Path(c['train_output']).parent,'NEW_OUTPUT_NO_OLD_RUN_MUTATION')
    return True

def prior_admission(c,prior,check):
    config_contract(c,prior);failed=exact_json(prior['original_failed_request'],check)
    need(failed['request_id']==prior['original_request_id'] and failed['engineering_completed'] is False and failed['native_returncode']==-15
        and failed['cleanup']['owned_processes_remaining']==0 and failed['cleanup']['held_child']['reaped'] is True and failed['post']['active_compute_processes']==0,'ORIGINAL_FAIL_TERMINAL_RETAINED')
    CPU=exact_json(prior['original_saved29_CPU_result'],check);controller=exact_json(prior['original_saved29_CPU_controller'],check)
    need(CPU['passed'] is True and CPU['checked_state_files']==29 and CPU['unsafe_loader_fallback'] is False and CPU['original_failure_preserved'] is True and CPU['actual_updates']==0
        and CPU['initial']['named_SHA256']==c['initial_state']['named_SHA256'],'REAL_TYPED29_QUALIFICATION')
    need(controller['request_id']==prior['typed_CPU_request_id'] and controller['passed'] is True and controller['worker_returncode']==0
        and controller['post']['controller_device_FDs']==[] and controller['post']['owned_descendants_remaining']==0 and controller['held_cleanup']['reaped'] is True,'REAL_TYPED29_CONTROLLER_POST')
    for step in ['20','25']:
        row=CPU['checkpoints'][step];need(row['typed_checked_members']==14 and row['adapter_tensors']==row['optimizer_states']==504 and row['adapter_elements']==43646976
            and row['optimizer_step']==int(step) and row['trainer_cursor']['global_step']==int(step) and len(row['rank_RNG'])==8,'REAL_CHECKPOINT20_25_TYPED')
    reference=exact_json(c['pair_initial_reference'],check);need(reference['pair_id']==c['pair_id'] and reference['rank']==0 and reference['initial_state']==c['initial_state'],'SAME_DOMAIN_INITIAL_SOURCE')
    for rank,pin in sorted(c['continuous_reference'].items()):
        row=exact_json(pin,check);need(row['mode']=='fresh25' and row['arm']=='domain' and row['pair_id']==c['pair_id']
            and len(row['training_inputs'])==len(row['forward_inputs'])==len(row['losses'])==100,'REAL_ALL8_CONTINUOUS100_NOT_NEW_DATA')
    return dict(original_failed_request_preserved=True,typed29_original_PASS=True,own_checkpoint20_25=True,continuous_ranks=8,expected_resume_microbatches_per_rank=20,expected_new_updates_per_rank=5,
        initial_state_restored_on_resume=False,checkpoint20_native_restore_required=True,historical_optimizer_live_digest_available=False,original_rank0_missing_receipt_not_reconstructed=True)
