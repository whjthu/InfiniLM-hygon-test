"""New16 CPU original terminal metadata audit only; no processor/model/media/references loaded."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def require(ok,why):
    if not ok:raise ValueError(why)


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate evidence JSON key');result[key]=value
        return result
    def finite(value):
        parsed=float(value);require(math.isfinite(parsed),'Nonfinite evidence JSON number');return parsed
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:require(False,'Invalid evidence JSON constant'))


def read_raw(path,cap=1048576,digest=None):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'Unsafe CPU evidence file')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'CPU evidence changed')
    pin=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());require(digest is None or pin['SHA256']==digest,'CPU evidence SHA differs')
    return raw,pin


def read(path,cap=1048576,digest=None):
    raw,pin=read_raw(path,cap,digest);return strict_json(raw),pin


def main(a):
    require(re.fullmatch('[0-9a-f]{64}',a.source_pins_sha256),'Published runtime SOURCE_PINS SHA required')
    require(re.fullmatch('[0-9a-f]{40}',a.publication) and re.fullmatch('[0-9a-f]{32}',a.request),'Actual published/request identities invalid')
    source=a.run/'src';outcome,op=read(a.run/'outcome.json');intent,ip=read(a.run/'intent.json')
    require(intent['claim']==outcome['claim']==a.publication and intent['request_id']==outcome['request_id']==a.request
        and intent['epoch']==outcome['epoch']==a.epoch and intent['deadline']==outcome['deadline']==a.epoch+300
        and intent['GPU_cards']==outcome['GPU_cards']==0,'Original new CPU context differs')
    require(math.isfinite(a.epoch) and outcome['owner']==intent['owner'] and bool(intent['owner']['host'])
        and outcome['budget_origin']=='client_dispatch_epoch' and a.epoch<=outcome['received_at']<a.epoch+250
        and 0<=outcome['wall_seconds']<300,'Original owner/whole dispatch budget differs')
    require(type(outcome['passed']) is bool,'Invalid original success type')
    child=outcome['child']
    require(outcome['group_RSS_threshold_bytes']==8589934592 and outcome['group_RSS_threshold_is_observed_not_hard_commit'] is True,
        'Original RSS threshold semantics differ')
    if child is not None:
        require(outcome['owned_group_reaped'] is True and child['PGID']==child['PID']
            and len(outcome['CPU_affinity'])==2 and not {0,1}&set(outcome['CPU_affinity']) and outcome['affinity_excludes_0_1'] is True,
            'Actual original CPU ownership/affinity boundary differs')
    else:
        require(outcome['passed'] is False and outcome['returncode'] is None and outcome['stop_reason'],
            'Missing child must be retained as actual prelaunch failure')
    result=dict(passed=True,producer_passed=outcome['passed'],planned_rows=16,completed_rows=0,actual_native_inference_rows=0,
        original_request=a.request,source_publication=a.publication,epoch=a.epoch,whole_deadline=a.epoch+300,
        original_returncode=outcome['returncode'],original_stop_reason=outcome['stop_reason'],owned_group_reaped=outcome['owned_group_reaped'],
        actual_child_started=child is not None,
        actual_CPU_affinity=outcome['CPU_affinity'],observed_peak_group_RSS_bytes=outcome['peak_group_RSS_bytes'],
        group_RSS_stop_threshold_bytes=8589934592,RSS_is_observed_not_hard_commit=True,
        Dataset_GO=False,training_GO=False,GPU_GO=False,model_loaded=None,scorer_calls=0,training_updates=0,
        new_processor_calls=0,original80_reference_captions_accessed=False,
        pins=dict(outcome=op,intent=ip))
    cpu_path=a.run/'cpu/CPU-result.json'
    require(not outcome['passed'] or (child is not None and cpu_path.is_file()),'Success missing original child/CPU receipt')
    if cpu_path.exists():
        cpu,cp=read(cpu_path,digest=outcome['CPU_result_SHA256']);require(cpu==outcome['CPU_result'],'Actual embedded CPU receipt differs')
        require(cpu['epoch']==a.epoch and cpu['deadline']==a.epoch+260 and cpu['planned_rows']==16
            and cpu['actual_CPU_affinity']==outcome['CPU_affinity'] and cpu['owner']['PID']==outcome['child']['PID']
            and cpu['owner']['startticks']==outcome['child']['startticks'],'Actual child CPU ownership/epoch differs')
        progress,pp=read(a.run/'cpu/progress.json');require(progress['planned_rows']==16 and progress['completed_rows']==cpu['completed_rows'],'Full progress denominator differs')
        result['pins'].update(CPU_result=cp,progress=pp);result['completed_rows']=cpu['completed_rows']
        require(type(cpu['completed_rows']) is int and 0<=cpu['completed_rows']<=16,'Invalid completed16 denominator')
        if outcome['passed'] is True:
            source_pins,spp=read(source/'SOURCE_PINS.json',digest=a.source_pins_sha256)
            require({p.name for p in source.iterdir()}==set(source_pins)|{'SOURCE_PINS.json'},'Runtime source inventory differs')
            for name,pin in source_pins.items():
                raw,actual=read_raw(source/name,cap=131072,digest=pin['SHA256']);require(actual==pin,'Runtime source identity differs')
            result['pins']['runtime_SOURCE_PINS']=spp
            require(outcome['returncode']==0 and outcome['stop_reason'] is None and cpu['passed'] is True and cpu['completed_rows']==16
                and cpu['actual_mode']=='transformers' and cpu['actual_template_type']=='llava_onevision1_5'
                and cpu['max_length']==4096 and cpu['max_pixels']==262144 and cpu['labels_None_rows']==cpu['full_rows_retained']==16,
                'Full16 new inference CPU success boundaries differ')
            require(cpu['metadata_files_verified']==14 and cpu['Swift_provider_files_verified']==7
                and cpu['framework_files']==350 and cpu['framework_commit']=='b641edb0ad0f5104f18ba4ab45a247fbe1f7c495'
                and cpu['new_model_view_exact_metadata_files']==14 and cpu['weight_files_in_model_view']==0
                and cpu['weight_payload_bytes_read']==0 and cpu['weight_guard_expected_denied']==5
                and cpu['unexpected_weight_open_or_load_attempts']==cpu['network_attempts']==cpu['device_initialization_attempts']==[]
                and cpu['device_FDs_before']==cpu['device_FDs_after']==[] and cpu['CUDA_initialized'] is False
                and cpu['model_loaded'] is False and cpu['original_model_loaded'] is False
                and cpu['original64_train_mode_qualification_inherited'] is False,
                'No-weight/device/new-scope qualifications differ')
            identities,rp=read(a.run/'cpu/input-identities-private.json',digest=cpu['identities_SHA256'])
            require(len(identities)==16,'Actual16 identity denominator differs')
            manifest,dp=read(a.run/'data/DATA_MANIFEST.json',digest='5b758107194df4f2599cad3d2598fb4f012eb220c3a75b1052db4d64eac3fcd1')
            tokens=[]
            for i,row in enumerate(identities):
                source_image=manifest['images'][i]
                require(row['source_image']=={k:source_image[k] for k in ('name','bytes','SHA256')}
                    and row['input_id']=='coco:'+str(int(Path(source_image['name']).stem)),'Native output/source image slot binding differs')
                require(type(row['sequence_length']) is int and type(row['image_tokens']) is int,'Invalid native token counts')
                require(row['slot']==i and row['labels_None'] is True and row['full_input_retained'] is True
                    and 0<row['sequence_length']<=4096 and row['image_tokens']>0,'Actual inference occurrence boundary differs')
                for name in ('encoded_fields','collated_fields'):
                    require('input_ids' in row[name] and 'pixel_values' in row[name]
                        and all(v['finite'] is True and re.fullmatch('[0-9a-f]{64}',v['SHA256']) for v in row[name].values()),
                        'Actual fields missing/nonfinite/unpinned')
                tokens.append(row['sequence_length'])
            providers,np=read(a.run/'cpu/native-providers.json',digest=cpu['providers_SHA256'])
            originals,osp=read(source/'ORIGINAL_SOURCES.json');allowed={v['SHA256'] for v in originals['files'].values()}
            require(set(providers)=={'encode','collator','set_mode'} and all(v['file_SHA256'] in allowed
                and re.fullmatch('[0-9a-f]{64}',v['method_source_SHA256']) for v in providers.values()),'Bound native provider identities differ')
            require(manifest['planned_rows']==len(manifest['images'])==16 and manifest['image_bytes']==2492125
                and manifest['labels_in_inputs'] is False and manifest['references_read']==0,'Frozen input source/answer boundary differs')
            metadata,mp=read(source/'MODEL_METADATA.json');view=a.run/'model-metadata-only'
            require({p.name for p in view.iterdir()}==set(metadata['files']) and len(metadata['files'])==14,'Exact metadata-only view inventory differs')
            for name,pin in metadata['files'].items():
                path=view/name;require(path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes']
                    and hashlib.sha256(path.read_bytes()).hexdigest()==pin['SHA256'],'Actual metadata-only view bytes differ')
            result.update(actual_native_inference_rows=16,missing_rows=0,token_range=[min(tokens),max(tokens)],labels_None_rows=16,
                model_loaded=False,
                new_T04_inference_CPU_qualified=True,original_T04_train64_qualification_inherited=False,
                new_physical_metadata_view_original14_bytes=True,photographer_attribution='PENDING',
                independent_reader_did_not_read_media_or_inputs_or_processed_values=True,
                device_boundary='Producer Python audit/load API guards and FD snapshots; no claim of kernel/native-C IO isolation',
                terminal_boundary='Original owned-group supervisor receipt; independent reader performs no fresh process query')
            result['pins'].update(identities=rp,native_providers=np,original_Swift_sources=osp,data_manifest=dp,model_metadata=mp)
    result.setdefault('missing_rows',16-result['actual_native_inference_rows'])
    require(not a.output.exists(),'Independent output already exists')
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('passed','producer_passed','planned_rows','completed_rows','actual_native_inference_rows','missing_rows','Dataset_GO','training_GO','GPU_GO')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--publication',required=True)
    p.add_argument('--request',required=True);p.add_argument('--source-pins-sha256',required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
