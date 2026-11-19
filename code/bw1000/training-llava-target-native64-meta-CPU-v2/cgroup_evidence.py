"""Independent validation of saved metadata; no live cgroup/Native/body reads."""
import math
from pathlib import PurePosixPath

def need(ok,why):
    if not ok:raise ValueError(why)

def validate(values,actual_step,epoch,worker_owner,request):
    a=values['cgroup-admission.json'];ready=values['native-ready.json'];release=values['native-release.json']
    v=values['container-control/outcome.json'];r=values['container-control/resources.json']
    identities=values['container-control/identity-observations.json'];registry=values['container-control/owned-registry.json']
    parent,step=actual_step.split('.')
    need(parent=='1087656' and step.isdigit(),'ACTUAL_CGROUP_NUMERIC_STEP')
    need(a['qualified'] is True and a['step']==actual_step and a['parent']==parent
         and a['epoch']==epoch and a['deadline']==epoch+300,'CGROUP_ADMISSION_BINDING')
    need(a['memory_controller_version'] in (1,2) and a['scope_is_numeric_step_not_job'] is True
         and a['charged_memory_is_sum_RSS'] is a['allocation_is_hard_limit'] is a['memory_control_files_written'] is False,
         'EXPLICIT_CHARGED_MEMORY_SCOPE')
    scope=PurePosixPath(a['hierarchy_scope']);parts=scope.parts
    need(scope.is_absolute() and parts[-2:]==('job_'+parent,'step_'+step),'EXACT_STEP_NOT_PARENT_OR_EXTERN')
    mount=a['mount'];mroot=PurePosixPath(mount['root'])
    need(mroot==scope or mroot in scope.parents,'MOUNT_ROOT_STEP_MAPPING')
    need(str(PurePosixPath(mount['mount'])/scope.relative_to(mroot))==a['physical_scope'],'CGROUP_PHYSICAL_MAPPING')
    need(mount['fs']==('cgroup2' if a['memory_controller_version']==2 else 'cgroup')
         and (a['memory_controller_version']==2 or 'memory' in mount['options']),'ACTUAL_MEMORY_CONTROLLER')
    owner=v['owner'];need(owner['PID']==a['self_PID'] and a['initial_sample']['member_PIDs']==[owner['PID']],
                         'INITIAL_SCOPE_EXCLUSIVITY')
    need(r==v['resources'] and r['metric']=='STEP_CGROUP_CHARGED_MEMORY'
         and r['soft_limit_bytes']==8*1024**3 and r['hard_limit'] is False
         and r['charged_memory_is_sum_RSS'] is False and r['RSS_group_bytes'] is None
         and r['RSS_coverage']=='UNKNOWN_NOT_READ','MEMORY_NOT_RSS_OR_HARD_LIMIT')
    samples=r['charged_samples'];need(0<len(samples)==r['samples']<=1200,'CHARGED_SAMPLE_DENOMINATOR')
    prior=-1.;known={int(k) for k in registry}|{owner['PID']}
    for s in samples:
        need(type(s['observed_monotonic']) in (int,float) and math.isfinite(s['observed_monotonic'])
             and s['observed_monotonic']>=prior,'MEMORY_SAME_HOST_SAMPLE_CLOCK')
        prior=s['observed_monotonic']
        need(type(s['charged_memory_bytes']) is int and 0<=s['charged_memory_bytes']<=8*1024**3
             and s['soft_limit_exceeded'] is False and set(s['member_PIDs'])<=known,'ACTUAL_CHARGED_SAMPLES')
        need(isinstance(s['events'],dict) and all(type(x) is int and x>=0 for x in s['events'].values()),'MEMORY_EVENTS')
        need(s['observed_limit']=='max' or type(s['observed_limit']) is int and s['observed_limit']>=0,'OBSERVED_LIMIT_ONLY')
    need(r['peak_charged_memory_bytes']==max(s['charged_memory_bytes'] for s in samples)
         and r['soft_limit_exceeded'] is False,'INDEPENDENT_CHARGED_PEAK')
    need(release['request_id']==request,'ORIGINAL_NATIVE_RELEASE_REQUEST')
    need(release['native_release_authorized'] is release['memory_scope_qualified'] is v['native_released'] is True,
         'HOST_MEMORY_GATE_BEFORE_NATIVE')
    need(release['epoch']==ready['epoch']==epoch and release['deadline']==ready['deadline']==epoch+300
         and release['work_deadline']==ready['work_deadline']==worker_owner['work_deadline']<=epoch+210,
         'ORIGINAL_RELEASE_DEADLINES')
    w=ready['owner'];need(release['owner']==w and ready['native_imports']==0 and ready['Torch_imported'] is False,
                        'EXACT_STD_LIB_HANDSHAKE')
    need(all(w[k]==worker_owner['owner'][k] for k in ('PID','PGID','startticks','UID','host','boot_id')),
         'SAME_RELEASED_NATIVE_PROCESS')
    need(all(release['memory_scope'][k]==a[k] for k in release['memory_scope']),'RELEASED_SCOPE_PROOF')
    need(any(s['observed_monotonic']==release['sample_monotonic'] and w['PID'] in s['member_PIDs'] for s in samples),
         'RELEASE_FOLLOWS_ACTUAL_MEMBER_SAMPLE')
    need(identities['RSS_coverage']=='UNKNOWN_NOT_READ' and identities['global_proc_scan'] is False
         and identities['total_records']>=len(identities['retained_records'])>0,'IDENTITY_EVIDENCE_DENOMINATOR')
    for row in identities['retained_records']:
        if row.get('vanished_during_observation'):
            need(row['accepted'] is False and row['original_PID_termination_proven'] is False
                 and row['error']['type'] in ('FileNotFoundError','ProcessLookupError'), 'EXPLICIT_PROC_DISAPPEARANCE_ONLY')
            continue
        need(row['accepted'] is True and row['RSS_bytes'] is None and row['RSS_coverage']=='UNKNOWN_NOT_READ',
             'STRICT_IDENTITY_NO_PRIVILEGED_RSS')
        uid=owner['UID'];klass=row['UID_classification']
        need((klass=='ORDINARY_OWNED' and row['UID_values']==[uid]*4 and row['proc_directory']['UID']==uid)
             or (klass=='ANCHORED_359_PRIVILEGED_INTERMEDIARY' and row['UID_values']==[uid,0,0,uid]
                 and row['proc_directory']['UID']==0),'ONLY_REVIEWED_UID_PATTERNS')
        need(all(row['before'][k]==row['middle'][k]==row['after'][k] for k in ('PID','PPID','PGID','startticks')),
             'STABLE_ORIGINAL_PROCESS_IDENTITY')
        need(row['proc_directory']==row['proc_directory_after'],'STABLE_PROCESS_DIRECTORY')
        path=PurePosixPath(row['memory_membership']);need(path==scope or scope in path.parents,'KNOWN_MEMBER_SCOPE')
    post=v['post'];need(post['step_only_controller_remains'] is post['remaining_count_covers_privileged_members'] is True
         and post['cgroup_final']['member_PIDs']==[owner['PID']]
         and post['known_tree']['remaining_known_owned']==[] and post['known_tree']['unknown_owned_liveness']==[],
         'WHOLE_KNOWN_TREE_AND_STEP_POST_NOT_USER_ONLY')
    return dict(memory_controller_version=a['memory_controller_version'],charged_memory_samples=len(samples),
                peak_charged_memory_bytes=r['peak_charged_memory_bytes'],RSS_coverage='UNKNOWN_NOT_READ',
                numerical_qualification=False,metadata_only=True)
