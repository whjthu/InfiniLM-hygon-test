"""Independent stdlib-only reader; original terminal accounting is required first."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct

LABEL='training-treasury-timeseries-native-CPU-package-v1'
PUBLICATION='99c1e7fda254da6a36042c320bd245f7db4ddfdf'
CLAIM_SHA='6319b2fd24a5980a50b35dbd6cae3379d9e4e01c05dc16e49e7f727a1804818f'
MF_SHA='e7e6f9a1add5dec14a56d7cf570c26d6adabced92fc81e6db32735bd58b1cc39'
DATA_SHA='6d0ce57a02fbfb5174d61ee3548c3b43bdcae1ddb4e5e6d589a7706dfb83ca5f'
NATIVE_MF_SHA='d4b802d666d4a72305400fdbabccf5d93611ca8e07c26f7c06a94b18be6d6035'
ORIGINAL_REQUEST='7971fa412cfa4c048898f40a31ee7b79'
LOCAL_DISPATCH_SHA='289a0322b37f02e9e6bc76e5a6e8e452209965e8be12c08a2b2c995f0e181057'
ORIGINAL_PINS={
    'intent.json':'b6b819e21946affd6760c4bf5f0c88c4422fc3d0043525a92f67ce4a131aae1d',
    'outcome.json':'aec6611c2bf9e94a1c9006acf69365e632788f33674b9e1ec071792a4298e895',
    'work/CPU-receipt.json':'e34ad26148c6ff77449d5021f957572af877f0ba4be53149c8d50c324027db84',
    'work/native/progress.json':'763851a1b7b39619c7d745f70585e4062ba9c661568c611b9a568ac16c714c83'}
COLUMNS='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
SPLITS=('train','dev','blind')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def require(ok,why):
    if not ok:raise ValueError(why)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def raw(path,limit=262144,expected=None):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=limit,'unsafe original file type/size')
    data=path.read_bytes();after=path.lstat()
    require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),
            'original changed while reading')
    identity=dict(bytes=len(data),SHA256=digest(data))
    if expected is not None:
        require(identity==expected if isinstance(expected,dict) else identity['SHA256']==expected,'original SHA mismatch')
    return data,identity


def read(path,limit=262144,expected=None):
    data,identity=raw(path,limit,expected)
    return json.loads(data),identity


def relative(name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'unsafe original member path')
    return p


def choose_row(capture,parent,label):
    require(capture['accounting_columns']==COLUMNS,'independent accounting columns differ')
    rows=capture['parents'][parent]
    require(isinstance(rows,list) and all(isinstance(row,list) and len(row)==7 for row in rows),
            'independent accounting row schema differs')
    matches=[row for row in rows if row[1]==label]
    require(len(matches)<=1,'duplicate independent original label')
    row=matches[0] if matches else None
    if row:require(re.fullmatch(parent+r'\.[0-9]+',row[0]),'independent handle has wrong parent')
    return row


def terminal(row):return bool(row and row[2].split()[0] in TERMINAL)


def resources(row):
    require(row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n11','actual CPU step failed or wrong node')
    require(row[4].isdigit(),'actual elapsed seconds missing')
    tres=dict(x.split('=',1) for x in row[6].split(','))
    require(tres.get('cpu')=='8' and tres.get('node')=='1' and tres.get('mem') in ('16G','16384M'),
            'actual CPU/memory/node allocation differs')
    require(all(value in ('0','none') for name,value in tres.items() if name.startswith('gres')),
            'actual GRES allocated')
    return dict(CPU_cores=8,memory_MiB=16384,GRES=0,elapsed_seconds=int(row[4]))


def package(run):
    source=run/'source'
    manifest,mfpin=read(source/'MF.json',expected=MF_SHA)
    claim,claimpin=read(source/'CLAIM.json',expected=CLAIM_SHA)
    expected=dict(claim['source_SHA256']);expected['CLAIM.json']=CLAIM_SHA
    require(set(manifest)==set(expected) and len(manifest)==17,'source manifest denominator differs')
    actual=set()
    for i,path in enumerate(source.rglob('*')):
        require(i<64 and not path.is_symlink(),'unsafe source traversal')
        if path.is_file():actual.add(path.relative_to(source).as_posix())
    require(actual==set(expected)|{'MF.json'},'18 source view differs')
    for name,identity in manifest.items():
        require(identity['SHA256']==expected[name],'claim/member SHA binding differs')
        raw(source/relative(name),expected=identity)
    require((claim['parent'],claim['node'],claim['CPU_cores'],claim['memory_MiB'],claim['GPU_cards'],
             claim['whole_epoch_seconds'],claim['Slurm_maximum_seconds'],claim['inside_maximum_seconds'])==
            ('1103606','f13r1n11',8,16384,0,600,480,450),'original resource/epoch claim differs')
    native,_=read(source/'native/MF.json',expected=NATIVE_MF_SHA)
    for name,identity in native.items():raw(source/'native'/relative(name),expected=identity)
    runtime,runtimepin=read(source/'RUNTIME_PINS.json')
    return runtime,dict(files=18,MF=mfpin,CLAIM=claimpin,native_copies=6,RUNTIME_PINS=runtimepin)


def data(run):
    root=run/'data';manifest,mpin=read(root/'manifest.json',expected=DATA_SHA)
    pins=manifest['files_SHA256']
    images={name for name in pins if name.endswith('.png')}
    expected={'manifest.json','train/labels.jsonl',*[s+'/inputs.jsonl' for s in SPLITS],*images}
    require(len(images)==48 and len(expected)==53,'frozen data denominator differs')
    actual=set()
    for i,path in enumerate(root.rglob('*')):
        require(i<80 and not path.is_symlink(),'unsafe data traversal')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    require(actual==expected,'53 data view differs or held-out gold installed')
    total=mpin['bytes'];sizes={};seen_ids=set();windows=set();media_shas=set();train_rows=None
    for split in SPLITS:
        proof=manifest['splits'][split]
        require((proof['examples'],proof['media'],proof['windows'],proof['raw_dates'])==(144,16,8,40),
                'original data split denominator differs')
        name=split+'/inputs.jsonl';payload,identity=raw(root/name,expected=pins[name]);total+=identity['bytes']
        rows=[json.loads(line) for line in payload.splitlines()]
        require(len(rows)==144 and Counter(r['topic'] for r in rows)==
                {'lookup':48,'signed_difference':48,'maximum_date':48},'input rows/topic denominator differs')
        split_windows={r['window_id'] for r in rows}
        require(len(split_windows)==8 and not split_windows&windows,'input window split overlap')
        windows.update(split_windows)
        for row in rows:
            require(row['id'] not in seen_ids,'duplicate input identity');seen_ids.add(row['id'])
            require([x['role'] for x in row['messages']]==(['user','assistant'] if split=='train' else ['user']),
                    'training supervision absent or held-out answer leaked')
            require(len(row['images'])==1 and row['images'][0] in images
                    and relative(row['images'][0]).parts[0]==split,'cross-split image reference')
        if split=='train':train_rows=rows
    labelraw,labelpin=raw(root/'train/labels.jsonl',expected=pins['train/labels.jsonl']);total+=labelpin['bytes']
    labels=[json.loads(line) for line in labelraw.splitlines()]
    require(len(labels)==144 and [(r['id'],r['answer']) for r in labels]==
            [(r['id'],r['messages'][1]['content']) for r in train_rows],'train labels/assistant mismatch')
    for name in images:
        payload,identity=raw(root/relative(name),limit=2*1024**2,expected=pins[name]);total+=identity['bytes']
        require(identity['SHA256'] not in media_shas,'shared media bytes');media_shas.add(identity['SHA256'])
        require(len(payload)>=24 and payload[:8]==b'\x89PNG\r\n\x1a\n','media is not original PNG')
        sizes[name]=dict(original_size=list(struct.unpack('>II',payload[16:24])),**identity)
    require(total==3489522 and len(seen_ids)==432 and len(windows)==24,'data total identity differs')
    return dict(files=53,bytes=total,rows={s:144 for s in SPLITS},windows=24,media_SHA_verified=48,
        manifest=mpin,dev_blind_gold_present=False,dev_blind_gold_opened=False,
        labels_bytes_SHA_verified={'train':True,'dev':False,'blind':False},
        label_SHA_references={s:pins[s+'/labels.jsonl'] for s in SPLITS}),sizes


def epoch_records(state,intent,outcome,owner,receipt,row):
    require(state['publication']==intent['publication']==PUBLICATION
            and state['claim_SHA256']==intent['claim_SHA256']==CLAIM_SHA,'publication/claim binding differs')
    epoch,deadline=state['epoch'],state['deadline']
    require(all(type(x) in (int,float) and math.isfinite(x) for x in (epoch,deadline)) and deadline==epoch+600,
            'original single600 epoch differs')
    require(intent['epoch']==outcome['epoch']==epoch and intent['deadline']==outcome['deadline']==deadline
            and receipt['deadline']==owner['deadline']==deadline,'epoch reset across original receipts')
    budget=intent['budget'];argv=intent['command']
    require('--jobid=1103606' in argv and '--nodelist=f13r1n11' in argv and '--gres=none' in argv
            and '--cpus-per-task=8' in argv and '--mem=16384M' in argv,'actual argv resource differs')
    slurm=budget['slurm_seconds'];inside=budget['inside_seconds'];client=budget['client_seconds'];work=budget['work_deadline']
    require(type(slurm) is int and 60<=slurm<=480 and slurm%60==0 and type(inside) is int
            and 0<inside<=450 and inside+10<=slurm and inside<client<=inside+20
            and work<=deadline-75 and receipt['work_deadline']==owner['work_deadline']==work,
            'actual work/Slurm/client budget differs')
    require('--time='+budget['slurm_time'] in argv and budget['slurm_time']==f'00:{slurm//60:02d}:00'
            and '--kill-after=10s '+str(inside) in argv[-1] and '--deadline '+str(deadline) in argv[-1]
            and '--work-deadline '+str(work) in argv[-1],'actual shell/time argv binding differs')
    require(int(row[4])<=slurm+10 and owner['parent']=='1103606' and owner['node']=='f13r1n11'
            and row[0]=='1103606.'+owner['step'],'actual owner/elapsed differs')
    require(outcome['accounting_columns']==COLUMNS and outcome['accounting']['terminal_visible'] is True
            and outcome['accounting']['row']==row and outcome['current_step_outcome']=='TERMINAL'
            and outcome['rc']==0 and outcome['client_timeout'] is False
            and outcome['owned_client_cleanup_signals']==[] and outcome['receipt_errors']==[]
            and outcome['passed'] is True,'original collector did not pass')
    require(state['maximum_invocations']==intent['maximum_invocations']==1
            and state['replay_allowed']==intent['replay_allowed']==outcome['replay_allowed']==False,
            'one-invocation/no-replay binding differs')
    return dict(epoch=epoch,deadline=deadline,Slurm_seconds=slurm,inside_seconds=inside,
                client_seconds=client,work_deadline=work,original_collector_passed=True)


def native_record(native,receipt,owner,sizes,label_pins=None):
    require(native['state']=='NATIVE_CPU_INPUT_AND_TRAIN_COLLATOR_PASS' and receipt['passed'] is True,
            'actual native check did not pass')
    require(native['planned_rows']==native['completed_rows']==receipt['planned_rows']=={s:144 for s in SPLITS}
            and native['actual_native_encode_calls']==receipt['planned_native_encode_calls']==576
            and native['native_rows']==432 and native['media_SHA_verified']==48,'native completion denominator differs')
    require(native['max_length']==4096 and native['max_pixels']==262144 and native['truncation']=='raise'
            and native['model_weights_loaded'] is False and native['rope_dummy_model_on_meta'] is True
            and native['collated_train_rows']==2 and native['actual_GPU_cards']==native['actual_updates']==0,
            'actual native template/mask/meta/collator qualification differs')
    require(native['versions']=={'ms-swift':'4.4.2','transformers':'5.12.1','qwen-vl-utils':'0.0.14'}
            and native['manifest_SHA256']==DATA_SHA and native['dev_blind_label_content_opened'] is False,
            'native environment or no-gold identity differs')
    require(receipt['actual_GPU_cards']==receipt['actual_updates']==0 and receipt['weight_open_attempt_hashes']==[]
            and receipt['model_weight_values_read'] is False and receipt['failure_type'] is None,
            'weight audit/CPU-only failure')
    require(receipt['fd_before']==owner['fd_before'] and receipt['full_fd_inventory_performed'] is True,
            'original own FD inventory binding differs')
    for key in ('fd_before','fd_after'):
        fd=receipt[key]
        require(fd['full_fd_inventory'] is True and type(fd['fd_count']) is int and fd['fd_count']>=0
                and fd['KFD_DRM_holders']==[],'actual own CPU FD inventory incomplete or KFD nonzero')
    summary={}
    for split in SPLITS:
        row=native['splits'][split]
        lengths=row['input_tokens_range'];tokens=row['image_tokens_range'];supervised=row['supervised_tokens']
        require(row['rows']==144 and len(lengths)==len(tokens)==2 and 0<lengths[0]<=lengths[1]<=4096
                and 0<tokens[0]<=tokens[1]<=lengths[1] and re.fullmatch('[0-9a-f]{64}',row['counts_SHA256']),
                'actual token/image range identity differs')
        require(type(supervised) is int and (144<=supervised<=144*4096 if split=='train' else supervised==0),
                'actual train/holdout supervision count differs')
        proof=native['label_identity_checks'][split]
        require(proof['actual_bytes_SHA256_verified']==proof['content_parsed']==(split=='train'),
                'native label audit reports held-out gold read or train unchecked')
        if label_pins is not None:require(proof['manifest_SHA256']==label_pins[split],'native label identity differs')
        summary[split]={k:row[k] for k in ('rows','input_tokens_range','image_tokens_range','supervised_tokens','counts_SHA256')}
    media=native['processed_media']
    require(set(media)==set(sizes) and len(media)==48,'native media completion differs')
    spatial=[]
    for name,info in media.items():
        require(all(info[key]==sizes[name][key] for key in ('original_size','bytes','SHA256'))
                and len(info['grid'])==3 and all(x>0 for x in info['grid'])
                and len(info['spatial_size'])==2 and all(x>0 for x in info['spatial_size'])
                and info['image_tokens']>0 and all(x>0 for x in info['pixel_values_shape']),
                'actual resized image/grid evidence differs')
        spatial.append(info['spatial_size'])
    return dict(splits=summary,actual_native_encode_calls=576,examples=432,media=48,
        collated_train_rows=2,producer_mask_and_collator_assertions_passed=True,
        per_token_payload_replayed=False,processed_width_range=[min(x[0] for x in spatial),max(x[0] for x in spatial)],
        processed_height_range=[min(x[1] for x in spatial),max(x[1] for x in spatial)],
        own_FD_before_after_KFD_DRM=0,semantic_readability='UNKNOWN')


def providers(root,run,runtime,receipt):
    require(len(runtime['processor'])==9,'processor metadata inventory differs')
    actual=set(path.name for path in (run/'model').iterdir())
    require(actual==set(runtime['processor']),'processor-only model view differs')
    for name,identity in runtime['processor'].items():raw(run/'model'/relative(name),limit=16*1024**2,expected=identity)
    prefix={
      'swift':('/swift-runtime/site/swift/',root/'results/training-vlm-swift-offline-runtime-v1/site'),
      'transformers':('/minicpm-transformers/transformers/',root/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site'),
      'qwen_vl_utils':('/vision/qwen_vl_utils/',root/'results/training-qwen-vision014-CPU-v2/vision'),
      'torch':('/usr/local/lib/python3.10/dist-packages/torch/',root/'containers/dtk26-sandbox-r1/usr/local/lib/python3.10/dist-packages')}
    require(set(receipt['actual_providers'])==set(prefix),'actual module provider count differs')
    for name,(start,host) in prefix.items():
        identity=receipt['actual_providers'][name];path=identity['path']
        require(path.startswith(start),'actual module provider prefix differs')
        container_root=start.rsplit(name+'/',1)[0]
        relative_name=path[len(container_root):]
        raw(host/relative(relative_name),limit=16*1024**2,expected={k:identity[k] for k in ('bytes','SHA256')})
    framework=root/'results/training-qwen-a68952c-swift-initial-CPU-v1/framework'
    for name,identity in runtime['framework'].items():raw(framework/relative(name),limit=16*1024**2,expected=identity)
    vision=root/'results/training-qwen-vision014-CPU-v2/vision'
    for name,identity in runtime['vision'].items():raw(vision/relative(name),expected=identity)
    raw(root/'results/training-vlm-swift-offline-runtime-v1/site/swift/template/templates/qwen.py',
        expected={k:runtime['template'][k] for k in ('bytes','SHA256')})
    raw(root/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site/transformers/models/qwen3_vl/modeling_qwen3_vl.py',
        expected={k:runtime['qwen_model'][k] for k in ('bytes','SHA256')})
    require(len(runtime['framework'])==393 and len(runtime['vision'])==7,'provider reused source denominator differs')
    return dict(processor_metadata_files=9,actual_module_providers_rehashed=4,framework_files_rehashed=393,
                vision_files_rehashed=7,weight_files_read=0,GPU_or_native_DSO_closure_qualified=False)


def original330(root,runtime,capture):
    base=root/'results/training-qwen-treasury-explicit-dev-CPU-v1'
    paths={'intent':'intent.json','outcome':'outcome.json','source_manifest':'source/SOURCE_FILES.json','log':'step.log'}
    for key,name in paths.items():raw(base/name,limit=4*1024**2,expected=runtime['CPU330'][key])
    row=choose_row(capture,'1087656','training-qwen-treasury-explicit-dev-CPU-v1')
    require(row and row[:4]==['1087656.330','training-qwen-treasury-explicit-dev-CPU-v1','COMPLETED','0:0']
            and row[5]=='a01r4n14','original CPU330 actual accounting differs')
    outcome,_=read(base/'outcome.json')
    require(outcome['passed'] is True and outcome['accounting']['row']==row,'original CPU330 outcome differs')
    members,_=read(base/'source/SOURCE_FILES.json')
    for name,identity in members['files'].items():raw(base/'source'/relative(name),expected=identity)
    return dict(handle='1087656.330',actual_accounting=row,original_artifacts_rehashed=4,
                reused_only_for_provider=True,new_dataset_qualified_by_old330=False)


def audit(root,capture):
    report=dict(state='ROOT_READER_NOT_TERMINAL_NO_ORIGINAL_READ',passed=False,
        original_publication=PUBLICATION,original_claim_SHA256=CLAIM_SHA,original_request=ORIGINAL_REQUEST,
        original_local_dispatch_SHA256=LOCAL_DISPATCH_SHA,planned_rows={s:144 for s in SPLITS},
        planned_examples=432,planned_native_encode_calls=576,planned_media=48,
        new_Slurm_steps=0,new_native_processor_calls=0,new_GPU_calls=0,new_updates=0,
        model_or_training_qualified=False,semantic_readability='UNKNOWN',formal17_cases_qualified=False,
        blind_effect_qualified=False,formal_performance_qualified=False,checks={},original_file_pins={})
    row=choose_row(capture,'1103606',LABEL)
    report['actual_accounting']=row
    if not terminal(row):return report
    run=root/'results'/LABEL
    report['state']='ROOT_READER_TERMINAL_AUDIT_FAILED'
    def section(name,fn):
        try:
            result=fn();report['checks'][name]=dict(passed=True,result=result);return result
        except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
            report['checks'][name]=dict(passed=False,error_type=type(error).__name__);return None
    section('actual_CPU_resources',lambda:resources(row))
    packaged=section('source18',lambda:package(run))
    bound=section('data53_noGold',lambda:data(run))
    documents={}
    for name,path in {'launch_state':'launch-state.json','intent':'intent.json','collector':'outcome.json',
                      'owner':'work/CPU-owner.json','receipt':'work/CPU-receipt.json','native':'work/native/progress.json'}.items():
        def load(path=path,name=name):
            value,identity=read(run/path,expected=ORIGINAL_PINS.get(path))
            require(isinstance(value,dict),'original receipt object differs')
            report['original_file_pins'][path]=identity;documents[name]=value
            return dict(original_present=True,pin=identity)
        section('document_'+name,load)
    if packaged:
        runtime=packaged[0]
        report['checks']['source18']['result']=packaged[1]
        section('original330_actual_reuse',lambda:original330(root,runtime,capture))
        if 'receipt' in documents:section('actual_provider_and_processor_identity',lambda:providers(root,run,runtime,documents['receipt']))
    required={'launch_state','intent','collector','owner','receipt','native'}
    if required<=set(documents):
        d=documents
        section('single_epoch_and_collector',lambda:epoch_records(d['launch_state'],d['intent'],d['collector'],d['owner'],d['receipt'],row))
        section('collector_native_receipt_identity',lambda:require(
            d['collector']['CPU_receipt']==d['receipt'] and d['collector']['native_progress']==d['native']
            and d['receipt']['native_progress_SHA256']==report['original_file_pins']['work/native/progress.json']['SHA256'],
            'collector/native original byte binding differs'))
        if bound:
            section('actual_native432_mask_collator_FD',lambda:native_record(
                d['native'],d['receipt'],d['owner'],bound[1],bound[0]['label_SHA_references']))
        report['actual_native_completed_rows']=d['native'].get('completed_rows')
        report['actual_native_failure_type']=d['native'].get('failure_type')
        report['actual_collector_passed']=d['collector'].get('passed')
    if bound:report['checks']['data53_noGold']['result']=bound[0]
    mandatory={'actual_CPU_resources','source18','data53_noGold','original330_actual_reuse',
               'actual_provider_and_processor_identity','single_epoch_and_collector',
               'collector_native_receipt_identity','actual_native432_mask_collator_FD'}
    report['passed']=mandatory<=set(report['checks']) and all(v['passed'] for v in report['checks'].values())
    if report['passed']:report['state']='ROOT_READER_NATIVE_CPU_INPUT_QUALIFICATION_PASS'
    report['qualification']='Recorded processor/mask/collator CPU qualification only; no per-token replay, semantic readability, model update or effect claim.'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--accounting',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    capture,accountingpin=read(a.accounting)
    result=audit(a.root,capture);result['independent_accounting_file_pin']=accountingpin
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],passed=result['passed'],planned_examples=432,
                         new_GPU_calls=0,new_native_processor_calls=0,semantic_readability='UNKNOWN')))
