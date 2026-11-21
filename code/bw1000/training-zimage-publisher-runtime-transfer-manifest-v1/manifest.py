"""Immutable publisher runtime manifest and bounded operation specs; no network."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

PUBLISHER=('results/bw1000/2026-11-22/training-zimage-and-internvl-publisher-byte-identities.json',
           '3ca66ba1344e2ec1ce2b9965446a7230705138aa8e579e0155fab419b29eaad9')
STATS=('results/bw1000/2026-11-22/training-zimage-four-original-A100-locators-checked-on-scnet.json',
       '65478875128e26239d1bf6b1b3e0f5e0fdada177776f99cddd4639a95fca1c09')
ORIGINAL=('coordination/training/zimage-original-complete-file-identity-v1.json',
          '07e5ead16f001edd585b50f48c5699302ff14311a060453e19eba3c21a7aee70')
PUBLICATION='30e954b9c4010bc91987c3af1d4acdd042bc5a3f'
SEGMENT=32*1024**2


def digest(raw):return hashlib.sha256(raw).hexdigest()


def pinned(repo,spec):
    raw=(repo/spec[0]).read_bytes()
    if digest(raw)!=spec[1]:raise ValueError('Published source identity differs')
    return json.loads(raw)


def build(repo):
    publisher=pinned(repo,PUBLISHER);original=pinned(repo,ORIGINAL);stats=pinned(repo,STATS)
    if (publisher['required_runtime_files']!=19 or len(publisher['zimage'])!=19
            or publisher['downloads_of_tensor_payloads']!=0
            or stats['group_states']!={k:'ENOENT' for k in ('base','preset','same_input','first_capture_progress')}):
        raise ValueError('Publisher/source-only qualification differs')
    base='https://huggingface.co/Tongyi-MAI/Z-Image-Turbo'
    preset='https://huggingface.co/ostris/zimage_turbo_training_adapter'
    revisions={'base':publisher['base_revision'],'preset':publisher['preset_revision']}
    if (publisher['publishers']['zimage']!=base+'/tree/'+revisions['base']
            or publisher['publishers']['preset']!=preset+'/blob/'+revisions['preset']+'/'+publisher['preset']['name']
            or any(re.fullmatch('[0-9a-f]{40}',v) is None for v in revisions.values())):
        raise ValueError('Exact publisher repository/revision differs')
    assets=[]
    sources=[('base',name,row) for name,row in sorted(publisher['zimage'].items())]
    sources.append(('preset',publisher['preset']['name'],publisher['preset']))
    for ordinal,(kind,name,row) in enumerate(sources,1):
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe publisher runtime path')
        original_pin=original['model' if kind=='base' else 'preset']['files'][name]
        size,sha=row['expected_bytes'],row['expected_SHA256']
        if (row['state'] not in ('PUBLISHER_BODY_SHA_AND_SIZE_MATCH','LFS_SHA_AND_SIZE_MATCH')
                or row['publisher_bytes']!=size or row['publisher_SHA256']!=sha
                or original_pin['bytes']!=size or original_pin['SHA256']!=sha):
            raise ValueError('Runtime component differs from frozen A100/source')
        assets.append({'asset_id':ordinal,'kind':kind,'relative_path':name,'filename':path.name,
                       'publisher_URL':(base if kind=='base' else preset)+'/resolve/'+revisions[kind]+'/'+name,
                       'revision':revisions[kind],'bytes':size,'SHA256':sha,
                       'publisher_proof':row['state'],'tensor_payload':name.endswith('.safetensors'),
                       'target_relative_path':kind+'/'+name,'segment_bytes':SEGMENT,
                       'planned_ranges':(size+SEGMENT-1)//SEGMENT})
    omitted={n:p for n,p in original['model']['files'].items() if n not in publisher['zimage']}
    if len(omitted)!=58:raise ValueError('Runtime subset/full77 receipt distinction differs')
    subset={v['target_relative_path']:{k:v[k] for k in ('bytes','SHA256')} for v in assets}
    return {'schema':'ZIMAGE_IMMUTABLE_PUBLIC_RUNTIME_SUBSET_TRANSFER_V1','source_publication':PUBLICATION,
            'publisher_evidence':{'path':PUBLISHER[0],'SHA256':PUBLISHER[1]},
            'successful_target_inventory_evidence':{'path':STATS[0],'SHA256':STATS[1],
              'meaning':'four original A100 root locators absent;101declared asset slots; no101file body qualification'},
            'assets':assets,'runtime_base_files':19,'preset_files':1,
            'total_runtime_bytes':sum(v['bytes'] for v in assets),
            'publisher_base_body_SHA_verified_files':11,'publisher_base_LFS_metadata_verified_files':8,
            'publisher_preset_LFS_metadata_verified_files':1,'publisher_tensor_body_bytes_read':0,
            'runtime_subset_manifest_SHA256':digest(json.dumps(subset,sort_keys=True,separators=(',',':')).encode()),
            'original77_receipt_not_redefined':True,'omitted_original_base_receipt_files':58,
            'omitted_source_sha_sidecars':sum(n.endswith('.source-sha256') for n in omitted),
            'omitted_other_original_receipt_files':sum(not n.endswith('.source-sha256') for n in omitted),
            'target_asset_ROOT':None,'target_provisioned':False,'target_wholefile_SHA_verified':False,
            'original_COCO20_or_initial_capture_included':False,'training_READY':False}


def validate(value,repo):
    if value!=build(repo):raise ValueError('Immutable runtime transfer manifest changed')
    return value


def operation(value,asset_id,offset):
    found=[v for v in value['assets'] if v['asset_id']==asset_id]
    if len(found)!=1 or type(offset) is not int:raise ValueError('Exact asset and integer offset required')
    row=found[0]
    if not 0<=offset<=row['bytes'] or (offset!=row['bytes'] and offset%SEGMENT):
        raise ValueError('Offset must be authoritative fixed-segment boundary')
    endpoint=urlsplit(row['publisher_URL'])
    if endpoint.scheme!='https' or endpoint.netloc!='huggingface.co' or endpoint.query:
        raise ValueError('Unexpected immutable publisher URL')
    count=min(SEGMENT,row['bytes']-offset)
    return {'schema':'ZIMAGE_ONE_CONFIRMED_OFFSET_OPERATION_V1','asset':row,
            'operation':'seal_only' if offset==row['bytes'] else 'fetch_one_range_after_target_inspection',
            'HTTP_headers':None if count==0 else {'Range':f'bytes={offset}-{offset+count-1}',
                                                'Accept-Encoding':'identity'},
            'required_Content_Range':None if count==0 else f'bytes {offset}-{offset+count-1}/{row["bytes"]}',
            'raw_bytes_max':count,'authoritative_target_confirmed_offset_required':True,
            'legacy_stream_source':'tools/stream_internvl_public_segment.py',
            'legacy_receiver_source':'tools/stage_internvl_weight_segment.py',
            'direct_legacy_call_allowed':False,
            'required_new_asset_bindings':['asset spec instead of InternVL-only shard_spec',
               'per-file immutable publisher parent URL/revision',
               'new target namespace/asset-id dirs and sealed-file runtime layout'],
            'reuse_protocol':'durable offset+segmentSHA+fsync; inspect before mutation;wholefileSHA seal',
            'local_weight_spool_allowed':False,'explicit_existing_transport_callback_required':True,
            'known_legacy_range_attempts_max':4,'legacy_window_seconds_max':420,
            'failure_rule':'Stop; fresh authoritative target inspection required; UNKNOWN never replayed',
            'execution_performed':False,'target_connection_available':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--build',action='store_true')
    p.add_argument('--asset-id',type=int);p.add_argument('--confirmed-offset',type=int)
    p.add_argument('--operation-output',type=Path)
    args=p.parse_args()
    if args.build:
        value=build(args.repo)
        with args.manifest.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    else:value=validate(json.loads(args.manifest.read_bytes()),args.repo)
    if args.asset_id is not None:
        if args.operation_output is None:raise ValueError('New operation output path required')
        spec=operation(value,args.asset_id,args.confirmed_offset)
        with args.operation_output.open('x',encoding='utf8') as f:json.dump(spec,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'assets':20,'bytes':value['total_runtime_bytes'],'validated':True,
                      'network_calls':0,'local_weight_writes':0,'training_READY':False},sort_keys=True))


if __name__=='__main__':main()
