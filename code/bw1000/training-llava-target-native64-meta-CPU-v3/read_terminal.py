"""Independent metadata-only consumer of the single new native64/metaPEFT CPU-v5 run."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-llava-target-native64-meta-CPU-v3')
EXPECTED_MANIFEST_SHA256='e55f2a28eed620a2b0712d43f96cc95293a84fcf2f339a899c81ca07ae298b3b'
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def need(ok,why):
    if not ok:raise ValueError(why)

def consume(context,read_claim):
    need(read_claim['root_GO'] is read_claim['READ_GO'] is True,'reader not adopted')
    PUB=context['publication'];SOURCE=context['source_manifest'];CLAIM=context['claim']
    need(len(PUB)==40 and not set(PUB)-set('0123456789abcdef'),'actual publication required')
    for p in (SOURCE,CLAIM):need(type(p['bytes']) is int and p['bytes']>0 and len(p['SHA256'])==64,'actual source/claim pin required')
    epoch=context['epoch'];request=context['request_id']
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<epoch<time.time()
         and len(request)==32 and not(set(request)-set('0123456789abcdef')),'original actual context malformed')
    read_epoch=context['read_epoch'];read_request=context['read_request_id']
    need(type(read_epoch) in (int,float) and math.isfinite(read_epoch) and 0<=time.time()-read_epoch<35,
         'fresh bounded reader epoch required')
    need(len(read_request)==32 and not set(read_request)-set('0123456789abcdef'),'reader request malformed')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,read_epoch+40-time.time())
    affinity=sorted(os.sched_getaffinity(0)-{0,1});need(len(affinity)>=2,'two reader CPUs excluding0/1 required')
    os.sched_setaffinity(0,affinity[:2]);resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    mono=time.monotonic()+read_epoch+40-time.time();total=0;pins={};values={};missing=[];errors=[]
    def load(name,optional=False):
        nonlocal total
        need(time.monotonic()<mono,'bounded metadata read exhausted')
        p=RUN/name
        try:before=p.lstat()
        except FileNotFoundError:
            if optional:missing.append(name);return None
            raise
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=2*1024**2,'regular bounded metadata member required')
        total+=before.st_size;need(total<=16*1024**2,'total metadata read cap')
        raw=p.read_bytes();after=p.lstat()
        need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'metadata changed during read')
        pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());values[name]=json.loads(raw);return values[name]
    mf=load('source/SOURCE_FILES.json');claim=load('source/CLAIM.json')
    need(pins['source/SOURCE_FILES.json']==SOURCE and pins['source/CLAIM.json']==CLAIM
        and mf['files']['CLAIM.json']==CLAIM,'exact CPU publication/MF/claim differs')
    installed_read_claim=load('source/TERMINAL_READ_CLAIM.json')
    need(pins['source/TERMINAL_READ_CLAIM.json']==mf['files']['TERMINAL_READ_CLAIM.json']
         and installed_read_claim==read_claim,'pre-registered reader claim differs from original source MF')
    plan=load('source/PLAN.json')
    need(pins['source/PLAN.json']==mf['files']['PLAN.json'] and
         claim['cache_alias_contract']==plan['cache_alias_contract'],'frozen cache alias contract')
    target_contract=load('source/TARGET_MATH_SOURCES.json')
    need(pins['source/TARGET_MATH_SOURCES.json']==mf['files']['TARGET_MATH_SOURCES.json']
         and claim['target_math_contract']['SHA256']==pins['source/TARGET_MATH_SOURCES.json']['SHA256']
         and target_contract['schema']=='T04_CUSTOM_DYNAMIC_EXACT_METADATA_WITH_MEASURED_TARGET_DEPENDENCIES_V1'
         and target_contract['numerical_equivalence_proven'] is False,'explicit target mathematics contract source pin')
    need(claim['root_GO'] is claim['CPU_GO'] is True and claim['GPU_GO'] is claim['Model_GO'] is claim['training_GO'] is False,'new CPU scope authority/limits')
    install=load('install-intent.json');intent=load('intent.json');out=load('outcome.json')
    for row in (install,intent,out):
        need(row['publication']==PUB and row['request_id']==request and row['epoch']==epoch and row['deadline']==epoch+300,'original request/epoch binding differs')
    need(install['MF_SHA256']==SOURCE['SHA256'] and intent['source_manifest']==SOURCE,'installed source context differs')
    need(out['accounting_columns']==COLS,'actual accounting columns absent/different')
    actual=out['accounting']['row']
    need(actual is not None and len(actual)==7 and actual[1]==RUN.name and actual[2].split()[0] in TERMINAL,'actual CPU step terminal not established')
    need(actual[0].startswith('1087656.') and actual[5]=='a14r3n14','original CPU parent/node differs')
    names=['step-intent.json','launcher-executable-identity.json','stage-before.json','container-intent.json','stage-outcome.json','worker-owner.json','worker-result.json',
           'caller-progress.json','native-caller-rows.json','native-caller-providers.json','native-PEFT-structure.json','encode-chain-observation.json','model-source-observation.json']
    for prefix in ('srun-control','container-control'):
        names += [prefix+'/'+n+'.json' for n in ('owner','held-child-anchor','child-owner','owned-registry','resources','outcome')]
    names += ['cgroup-admission.json','native-ready.json','native-release.json','container-control/identity-observations.json','cache-alias-observation.json']
    for name in names:load(name,optional=True)
    failure=load('failure-private.json',optional=True)
    # Never export the exception message/trace; exact private pin suffices for diagnosis.
    failure_summary={k:failure.get(k) for k in ('phase','error_type')} if failure else None
    if failure:
        message=failure.get('message','');need(type(message) is str and len(message)<=4096,'bounded original private failure message')
        failure_summary.update(message_SHA256=hashlib.sha256(message.encode()).hexdigest(),message_bytes=len(message.encode()),
            private_only_reason_last1024=message[-1024:],raw_reason_must_not_be_published=True)
    # A failure before the worker try-block has no failure-private.json. Inspect
    # only these two bounded log tails; never promote them to whole-file hashes.
    diagnostic_tails={};tail_pins={}
    if out.get('passed') is not True and failure is None:
        for name in ('container-control/stderr.private','srun-control/stderr.private'):
            need(time.monotonic()<mono,'reader log-tail deadline')
            p=RUN/name
            try:before=p.lstat()
            except FileNotFoundError:continue
            need(stat.S_ISREG(before.st_mode) and before.st_size<=16*1024**2,'bounded original private log')
            count=min(4096,before.st_size)
            if count==0:continue
            with p.open('rb') as stream:
                stream.seek(before.st_size-count);raw=stream.read(count);after=os.fstat(stream.fileno())
            key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            need(key(before)==key(after)==key(p.lstat()) and len(raw)==count,'original log changed')
            total+=count;need(total<=16*1024**2,'metadata plus log-tail cap')
            tail_pins[name]=dict(total_file_bytes=before.st_size,explicit_tail_bytes=count,
                tail_SHA256=hashlib.sha256(raw).hexdigest(),whole_file_SHA256_verified=False)
            diagnostic_tails[name]=raw.decode('utf-8',errors='replace')[-1024:]
    progress=values.get('caller-progress.json') or {};worker=values.get('worker-result.json') or {}
    completed=worker.get('completed_rows',progress.get('completed_rows',0))
    need(type(completed) is int and 0<=completed<=64,'actual completed/missing denominator invalid')
    owner=intent['owner'];need(owner['UID']==os.getuid(),'original controller numeric owner differs');host=socket.gethostname();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    owner_state='UNKNOWN_DIFFERENT_HOST_OR_BOOT'
    if host==owner['host'] and boot==owner['boot_id']:
        p=Path('/proc')/str(owner['PID'])/'stat'
        try:
            f=p.read_text().rsplit(') ',1)[1].split()
            owner_state='SAVED_OWNER_STILL_PRESENT' if int(f[19])==owner['startticks'] else 'SAVED_OWNER_ABSENT_PID_REUSED'
        except (FileNotFoundError,ProcessLookupError):owner_state='SAVED_OWNER_ABSENT'
    control_summaries={};native_summary=None;memory_summary=None
    for prefix in ('srun-control','container-control'):
        v=values.get(prefix+'/outcome.json')
        if v:
            r=v.get('resources',{});f=v.get('failure') or {}
            control_summaries[prefix]=dict(passed=v.get('passed'),returncode=v.get('returncode'),failure_type=f.get('type',f.get('error_type')),
                private_only_failure_reason=(v.get('failure') or {}).get('reason','')[:256],
                failure_reason_SHA256=hashlib.sha256((v.get('failure') or {}).get('reason','').encode()).hexdigest(),
                failure_phase=(v.get('failure') or {}).get('phase'),
                failure_source_linecode=(v.get('failure') or {}).get('source_linecode'),
                held_child_reaped=(v.get('held_cleanup') or {}).get('reaped'),held_termination_needed=(v.get('held_cleanup') or {}).get('termination_needed'),
                owned_processes_remaining=(v.get('registry_cleanup') or {}).get('owned_processes_remaining'),
                RSS_sampled_peak_bytes=r.get('sampled_group_peak_RSS_bytes') if prefix=='srun-control' else None,
                RSS_samples=r.get('samples') if prefix=='srun-control' else None,
                RSS_unknown_coverage_samples=r.get('RSS_unknown_coverage_samples',0) if prefix=='srun-control' else None,
                RSS_sampled_soft_limit_bytes=r.get('soft_limit_bytes') if prefix=='srun-control' else None,RSS_not_hard_limit=True,
                memory_metric=r.get('metric','OWNED_USER_RSS'),charged_memory_samples=r.get('samples') if prefix=='container-control' else None,
                peak_charged_memory_bytes=r.get('peak_charged_memory_bytes'),charged_memory_is_sum_RSS=False,
                native_released=v.get('native_released'),post=v.get('post'),cleanup_errors=v.get('cleanup_errors'))
    if out.get('passed') is True:
        try:
            need(not missing or missing==['failure-private.json'],'success missing required native/cleanup metadata')
            need(actual[2:4]==['COMPLETED','0:0'] and set(actual[6].split(','))=={'cpu=8','mem=12G','node=1'},'actual C0 CPU8/mem12G/0GRES required')
            need(out['actual_CPU_steps']==1 and out['actual_GPU_steps']==0 and out['execution_state']=='TERMINAL','one actual CPU0GPU step')
            exe=values['launcher-executable-identity.json'];need(exe['path']=='/public/software/apps/apptainer/1.3.4/bin/apptainer' and exe['X_OK'] is exe['regular'] is exe['identity_stable'] is True,'actual absolute executable stat/X_OK not established')
            stage=values['stage-outcome.json'];step=values['step-intent.json'];ci=values['container-intent.json'];wo=values['worker-owner.json']
            need(out['inside']==stage and stage['step']==actual[0] and stage['passed'] is True and stage['allocated_GPU_cards']==0,'actual stage binding')
            need(worker==stage['worker'] and worker['passed'] is True and worker['planned_rows']==worker['completed_rows']==64 and worker['missing_rows']==0,'full native64 actual result')
            need(progress==dict(planned_rows=64,completed_rows=64,missing_rows=0,failed=False),'full64 caller progress')
            for v in (stage,step,ci,worker,wo):need(v['epoch']==epoch and v['deadline']==epoch+300,'component epoch changed')
            need(ci['command'][0]=='/public/software/apps/apptainer/1.3.4/bin/apptainer' and '--rocm' not in ci['command'] and ['/usr/local/hyhal','/opt/hyhal','ro'] in ci['resolved_binds'],'qualified CPU launcher/systemalias plumbing differs')
            need(worker['work_deadline']<=epoch+210 and ci['work_deadline']==worker['work_deadline'] and ci['step_deadline']==step['step_deadline']<=epoch+255,'worker/Slurm deadlines')
            argv=step['command'];need('--gres=none' in argv and '-c8' in argv and '--mem=12288M' in argv and '-w' in argv and argv[argv.index('-w')+1]=='a14r3n14','actual CPU resource argv')
            need(step['Slurm_seconds'] in (120,180,240) and '--time=00:%02d:00'%(step['Slurm_seconds']//60) in argv,'shrinking actual Slurm budget')
            for prefix in ('srun-control','container-control'):
                v=values[prefix+'/outcome.json'];post=v['post'];held=v['held_cleanup'];tree=v['registry_cleanup']
                need(v['owner']==values[prefix+'/owner.json'] and v['child']==values[prefix+'/held-child-anchor.json'],'saved controller/held owner differs')
                child=values[prefix+'/child-owner.json'];need(v['child']['PID']==child['PID']==v['child']['PGID'] and child['startticks']>0,'actual held child identity')
                need(v['passed'] is True and v['returncode']==0 and v['failure'] is None and not v['cleanup_errors'],'actual bounded child C0/cleanup error')
                need(held['reaped'] is True and held['termination_needed'] is False and tree['owned_processes_remaining']==0 and tree['termination_needed'] is False,'held/tree normal completion required')
                need(post['same_host'] is post['same_boot'] is post['held_child_reaped'] is True and post['controller_device_FDs']==[]
                     and post['owned_descendants_remaining']==0 and post['owned_device_FD_holders']=={},'owned post not clean')
                r=v['resources'];need(r==values[prefix+'/resources.json'] and r['samples']>0 and r['soft_limit_bytes']==8*1024**3
                     and r['soft_limit_exceeded'] is False and r['hard_limit'] is False,'sampled resource limit qualification')
                need(epoch<v['stop']<v['held_deadline']<v['registry_deadline']<=epoch+282,'same-epoch owned cleanup deadlines')
                if prefix=='container-control':
                    helper_path=RUN/'source/cgroup_evidence.py'
                    need(helper_path.is_file() and not helper_path.is_symlink() and helper_path.stat().st_size==mf['files']['cgroup_evidence.py']['bytes']<=32768,'bounded predicate source')
                    need(total+mf['files']['cgroup_evidence.py']['bytes']<=16*1024**2,'metadata plus predicate source cap')
                    helper_raw=helper_path.read_bytes()
                    total+=len(helper_raw);need(total<=16*1024**2,'actual metadata plus predicate source cap')
                    pins['source/cgroup_evidence.py']=dict(bytes=len(helper_raw),SHA256=hashlib.sha256(helper_raw).hexdigest())
                    need(len(helper_raw)==mf['files']['cgroup_evidence.py']['bytes']<=32768 and hashlib.sha256(helper_raw).hexdigest()==mf['files']['cgroup_evidence.py']['SHA256'],'independent saved-memory predicate source pin')
                    helper_namespace={};exec(compile(helper_raw,str(helper_path),'exec'),helper_namespace)
                    memory_summary=helper_namespace['validate'](values,actual[0],epoch,wo,request)

            observed=values['model-source-observation.json']
            need(observed['schema']=='ACTUAL_TARGET_MATH_PROVIDER_OBSERVATION_V1' and observed['enumeration_complete'] is True and observed['validation_attempted'] is False and not observed['errors'] and observed['numerical_equivalence_proven'] is False,'complete target custom/library observation')
            need(observed['records'] and all(r['resolved_path']==r['resolved_module_path'] and r['module_file'] and r['inspect_sourcefile'] and 0<r['bytes']<=1024**2 and len(r['SHA256'])==64 for r in observed['records']),'actual target provider source coherence')
            custom=[r for r in observed['records'] if Path(r['resolved_path']).name==target_contract['custom_metadata_source']]
            need(custom and all({k:r[k] for k in ('bytes','SHA256')}==target_contract['custom_metadata_source_pin'] for r in custom),'exact custom14 mathematical code')
            rows=values['native-caller-rows.json'];structure=values['native-PEFT-structure.json'];providers=values['native-caller-providers.json']
            need(structure['actual_model_sources']==observed['records'] and structure['platform_source_contract_SHA256']==pins['source/TARGET_MATH_SOURCES.json']['SHA256'] and structure['reference_model_source_SHA256']==structure['measured_target_model_source_SHA256']==target_contract['reference_model_source_SHA256'] and structure['numerical_equivalence_proven'] is worker['numerical_equivalence_proven'] is False,'meta504 platform observation contract')
            need(len(rows)==64 and [r['ordinal'] for r in rows]==list(range(1,65)),'all64 ordered native rows')
            pre=load('source/INPUT_REFERENCE.json')['identities'];collator=load('source/COLLATOR_REFERENCE.json')['rows'];need(len(pre)==len(collator)==64,'same-phase A10064 public references')
            for i,row in enumerate(rows):
                need(0<row['sequence_length']<=4096 and row['supervised_EOS_count']>0 and row['effective_shifted_supervised_tokens']>0 and row['supervised_tokens']>0 and row['image_tokens']>0,'native full sequence/lossmask/EOS/images')
                need(row['selected_logit_rows']==row['sequence_length'] and row['native_logits_policy'] is False and row['post_model200_compared'] is False and all(row[k] is True for k in ('complete_input_sequence_unchanged','non_label_fields_unchanged','label_and_logit_mask_native_semantics_checked')),'T04 native full logits/no selection semantics')
                need(row['before_fields']==row['after_fields'] and 'logits_to_keep' not in row['before_fields'] and all(v['finite'] is True for v in row['before_fields'].values()),'native fields/labels unmodified')
                need(row['original_encoded_fields_exact']==(row['encoded_fields']==pre[i]['fields'] and row['sequence_length']==pre[i]['sequence_length']) and row['original_pre_model_collated_fields_exact']==(row['before_fields']==collator[i]['collated_fields'] and row['sequence_length']==collator[i]['sequence_length']),'independent same-phase fingerprint diagnostic')
            need(worker['pre_model64_reference_exact']==all(r['original_encoded_fields_exact'] for r in rows) and worker['pre_model64_collator_reference_exact']==all(r['original_pre_model_collated_fields_exact'] for r in rows),'diagnostic not promoted to native mathematical equality')
            need(worker['actual_trainable_tensors']==structure['actual_trainable_tensors']==len(structure['trainable_manifest'])==504 and worker['actual_trainable_parameters']==structure['actual_trainable_parameters']==sum(p['numel'] for p in structure['trainable_manifest'])==43646976,'meta504 real denominator')
            need(hashlib.sha256(json.dumps(structure['trainable_manifest'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==EXPECTED_MANIFEST_SHA256,'original504 exact names/shape/dtype')
            need(set(structure['frozen'])=={'model.visual'} and all(v['tensors']>0 and v['parameters']>0 for v in structure['frozen'].values()),'T04 actual visual frozen boundary')
            life=structure['lifecycle'];need(life['original_requested'] is life['language_checkpointing'] is life['vision_checkpointing'] is life['native_Trainer_flag'] is False,'T04 actual GCfalse')
            need(structure['initial_values_qualified'] is False and worker['actual_model_loaded'] is worker['training_forward_backward'] is False
                 and worker['implicit_library_device_queries']=='UNKNOWN' and worker['own_device_FDs_after']==[] and worker['weight_file_open_attempts']==[],'CPU only, no weights/FDs; implicit GPU remains unknown')
            chain=providers['encode']['wrapper_chain'];need(len(chain)==3 and providers==worker['providers'],'actual provider observations not result-bound')
            need([r['code_qualname'] for r in chain]==['context_decorator.<locals>.decorate_context','retry_decorator.<locals>._retry.<locals>.new_func','Template.encode'],'closed actual native wrapper chain')
            need(values['encode-chain-observation.json']['chain']==[{k:v for k,v in r.items() if k!='role'} for r in chain],'pre-caller actual wrapper observation differs')
            need(set(providers)=={'encode','collator','base_collator','logit_policy','logit_selection','native_prepare_inputs','post_encode','prepare_adapter','GC'},'complete actual method provider denominator')
            for r in [*chain,*[v for k,v in providers.items() if k!='encode']]:
                need(type(r['bytes']) is int and 0<r['bytes']<=262144 and len(r['SHA256'])==64 and len(r['definition_source_SHA256'])==64,'actual provider bounded full source identity')
            tokens=sum(r['effective_shifted_supervised_tokens'] for r in rows)
            need(tokens==worker['actual_shifted_supervised_tokens'] and worker['native_supervised_EOS_rows']==64,'actual native token/EOS aggregates')
            native_summary=dict(rows=64,supervised_EOS_rows=64,actual_shifted_supervised_tokens=tokens,sequence_length_range=[min(r['sequence_length'] for r in rows),max(r['sequence_length'] for r in rows)],
                actual_meta_trainable_tensors=504,actual_meta_trainable_parameters=43646976,provider_methods=9,actual_encode_chain=3,initial_values_qualified=False,pre_model64_encoded_exact=worker['pre_model64_reference_exact'],pre_model64_collator_exact=worker['pre_model64_collator_reference_exact'],post_model_comparison=False)
        except (KeyError,TypeError,ValueError,AssertionError) as e:errors.append(dict(type=type(e).__name__,reason=str(e)[:256]))
    observation=values.get('model-source-observation.json')
    observation_summary=None
    if isinstance(observation,dict):
        unique={}
        for row in observation.get('records',[]):
            keep={k:row.get(k) for k in ('prefix','module','module_file','inspect_sourcefile','resolved_path','resolved_module_path','bytes','SHA256')}
            unique[json.dumps(keep,sort_keys=True)]=keep
        observation_summary=dict(planned_prefixes=observation.get('planned_prefixes'),
            enumeration_complete=observation.get('enumeration_complete'),
            observed_class_count=len(observation.get('records',[])),
            source_identities=list(unique.values()),errors=observation.get('errors'),
            identities_are_observed_not_automatically_accepted=True,numerical_equivalence_proven=False)
    cache_alias=values.get('cache-alias-observation.json')
    alias_contract=plan['cache_alias_contract']
    alias_ok=cache_alias == dict(provider=alias_contract['provider'], model_SHA256=alias_contract['model_sha'],
             transformers_version=alias_contract['transformers_version'], static_cache_bound=True,
             temporary_export_restored=True)
    qualified=out.get('passed') is True and not errors and alias_ok
    result=dict(state='T04_TARGET_NATIVE64_META_PEFT_INDEPENDENT_PASS' if qualified else 'ORIGINAL_CPU_FAILURE_OR_INDEPENDENT_MISMATCH_RETAINED',
        independent_metadata_consumed=True,qualification_passed=qualified,producer_passed=out.get('passed'),publication=PUB,request_id=request,epoch=epoch,deadline=epoch+300,
        source_manifest=SOURCE,claim=CLAIM,actual_accounting_columns=COLS,actual_accounting=actual,planned_rows=64,completed_rows=completed,missing_rows=64-completed,
        private_failure=failure_summary,private_failure_pin=pins.get('failure-private.json'),missing_metadata=missing,errors=errors,original_HOST_owner_state=owner_state,
        private_only_diagnostic_tails=diagnostic_tails,diagnostic_tail_pins=tail_pins,
        raw_diagnostic_tails_must_not_be_published=True,reader_request_id=read_request,reader_epoch=read_epoch,
        controller_private_failure_reasons_must_not_be_published=True,current_login_liveness_required_for_software_qualification=False,
        controllers=control_summaries,native=native_summary,step_charged_memory=memory_summary,actual_metadata_files=len(pins),actual_metadata_read_bytes=total,metadata_pins=pins,
        weight_or_image_or_metadata14_or_framework393_body_reads=0,new_GPU_or_processor_calls=0,implicit_library_device_queries='UNKNOWN',
        cache_alias_observation=cache_alias,cache_alias_receipt_matches=alias_ok,math_provider_observation=observation_summary,target_math_contract_pin=pins['source/TARGET_MATH_SOURCES.json'],model_source_observation_pin=pins.get('model-source-observation.json'),numerical_equivalence_proven=False,
        initial_values_qualified=False,training_numerical_GO=False,GPU_GO=False,Model_GO=False,formal17_GO=False,replay_allowed=False)
    raw=json.dumps(result,sort_keys=True,allow_nan=False).encode();need(len(raw)<=16384,'bounded independent private summary cap')
    signal.setitimer(signal.ITIMER_REAL,0)
    return result

if __name__=='__main__':
    print(json.dumps(consume(CONTEXT,READER_CLAIM),sort_keys=True,allow_nan=False))
