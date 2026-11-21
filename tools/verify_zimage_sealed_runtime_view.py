"""Read sealed metadata and plan immutable file binds; never link/copy sealed assets."""
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

REPO=Path(__file__).resolve().parents[1]
MANIFEST='code/bw1000/training-zimage-publisher-runtime-transfer-manifest-v1/RUNTIME_FETCH_MANIFEST.json'
MANIFEST_SHA='fd2c299b1d6dccc36b021a22f7f3d285f926d19cb0e631c3d865c5e5212ca172'
STAGE=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-runtime-assets-v1')


def need(ok,why):
    if not ok:raise ValueError(why)


def spec(row):
    return dict(file_id=row['asset_id'],filename=row['filename'],bytes=row['bytes'],SHA256=row['SHA256'],
                segment_bytes=32*1024**2,kind=row['kind'],target_relative_path=row['target_relative_path'],
                publisher_URL=row['publisher_URL'])


def stamp(s):
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink)


def canonical_file(path):
    value=path.lstat()
    need(stat.S_ISREG(value.st_mode) and not path.is_symlink() and path.resolve(strict=True)==path,
         'Expected canonical regular file')
    return stamp(value)


def verify_leaf(row,stage_root):
    """No payload open, stage.lock open, hardlink creation or seal mutation."""
    result={'asset_id':row['asset_id'],'state':'UNOBSERVED','verified':False,'payload_bytes_read':0}
    try:
        root=stage_root/('asset-%02d'%row['asset_id'])
        checkpoint=root/'checkpoint.json'
        before=canonical_file(checkpoint)
        need(before['bytes']<=131072,'Oversized stage checkpoint')
        with checkpoint.open('rb') as f:raw=f.read(131073)
        need(len(raw)==before['bytes'] and canonical_file(checkpoint)==before,'Checkpoint changed during read')
        state=json.loads(raw)
        need(state['spec']==spec(row) and state['sealed'] is True and state['confirmed_bytes']==row['bytes'],
             'Asset is not fully sealed against this exact manifest')
        need(type(state['file']) is list and len(state['file'])==2
             and all(type(x) is int and x>=0 for x in state['file'])
             and type(state['sealed_stat']) is list and len(state['sealed_stat'])==2
             and all(type(x) is int and x>=0 for x in state['sealed_stat'])
             and type(state['segments']) is list,'Malformed sealed metadata anchors')
        consumed=0
        for entry in state['segments']:
            count=min(32*1024**2,row['bytes']-consumed)
            need(count>0 and type(entry['offset']) is int and entry['offset']==consumed
                 and type(entry['bytes']) is int and entry['bytes']==count
                 and re.fullmatch('[0-9a-f]{64}',entry['SHA256']) is not None,'Broken durable segment ledger')
            consumed+=count
        need(consumed==row['bytes'],'Incomplete segment denominator')
        leaf=root/row['filename'];a=canonical_file(leaf);b=canonical_file(root/'payload.partial')
        need(a==b and [a['device'],a['inode']]==state['file'] and a['bytes']==row['bytes']
             and [a['mtime_ns'],a['ctime_ns']]==state['sealed_stat'],'Sealed inode/mtime/ctime identity changed')
        need(canonical_file(checkpoint)==before and canonical_file(leaf)==a,'Seal changed during observation')
        result.update(state='SEALED_METADATA_AND_INODE_CONSISTENT',verified=True,leaf=str(leaf),
                      sealed_identity=a,checkpoint_SHA256=sha256(raw).hexdigest(),segments=len(state['segments']))
    except FileNotFoundError:result.update(state='ABSENT_OR_PARTIAL_RUNTIME',error_type='FileNotFoundError')
    except PermissionError:result.update(state='PERMISSION_UNKNOWN',error_type='PermissionError')
    except (ValueError,TypeError,KeyError) as e:
        result.update(state='SEAL_METADATA_REJECTED',error_type=type(e).__name__,reason=str(e)[:200])
    return result


def verify_runtime(manifest,stage_root,selected=None,mounted_root=None,require_readonly=True):
    rows=manifest['assets'];need(len(rows)==20 and {x['asset_id'] for x in rows}==set(range(1,21)),'Expected20 exact assets')
    ids=set(range(1,21)) if selected is None else set(selected)
    need(ids and ids<=set(range(1,21)),'Unknown asset selection')
    observed=[verify_leaf(row,stage_root) for row in rows if row['asset_id'] in ids]
    all20=len(observed)==20 and all(x['verified'] for x in observed)
    result=dict(state='ALL20_SEAL_METADATA_CONSISTENT' if all20 else 'PARTIAL_OR_UNQUALIFIED_RUNTIME',
                planned_assets=20,selected_assets=len(ids),verified_sealed_assets=sum(x['verified'] for x in observed),
                all20_sealed_metadata_consistent=all20,assets=observed,payload_bytes_read=0,
                independent_payload_SHA_verified=False,Model_GO=False,training_READY=False)
    if mounted_root is not None:
        need(all20,'No mounted view acceptance before all20 sealed assets')
        for row,source in zip(rows,observed,strict=True):
            target=mounted_root/row['target_relative_path']
            need(canonical_file(target)==source['sealed_identity'],'Mounted regular file does not expose exact sealed inode')
            if require_readonly:
                need(hasattr(os,'statvfs') and hasattr(os,'ST_RDONLY')
                     and os.statvfs(target).f_flag & os.ST_RDONLY,'Consumer file is not on a verified read-only mount')
            need(verify_leaf(row,stage_root)['sealed_identity']==source['sealed_identity'],'Source seal changed during view check')
        result.update(mounted_view_metadata_verified=True,mounted_model_root=str(mounted_root/'base'),
                      mounted_preset=str(mounted_root/'preset'/rows[-1]['filename']),
                      readonly_mount_verified=require_readonly)
    return result


def file_bind_plan(manifest,observation,stage_root,skeleton_root):
    """Only return a plan. Caller must create the fresh skeleton without touching stage."""
    need(observation['all20_sealed_metadata_consistent'] is True,'Full20 gate is closed')
    need(stage_root.is_absolute() and skeleton_root.is_absolute() and not skeleton_root.exists()
         and not skeleton_root.is_symlink(),'Fresh absolute skeleton root required')
    a=stage_root.resolve(strict=True);b=skeleton_root.resolve()
    need(b==skeleton_root and not b.is_relative_to(a) and not a.is_relative_to(b),
         'Model skeleton aliases or overlaps sealed source')
    target=PurePosixPath('/zimage-runtime-view')
    binds=[str(stage_root)+':/zimage-sealed-assets:ro',str(skeleton_root)+':'+str(target)+':ro']
    placeholders=[]
    for row in manifest['assets']:
        relative=PurePosixPath(row['target_relative_path'])
        need(not relative.is_absolute() and '..' not in relative.parts,'Unsafe runtime relative path')
        source=stage_root/('asset-%02d'%row['asset_id'])/row['filename']
        binds.append(str(source)+':'+str(target/relative)+':ro')
        placeholders.append({'relative_path':str(relative),'exclusive_empty_regular_file_bytes':0})
    return dict(schema='ZIMAGE_CANONICAL_READONLY_FILE_BIND_CONSUMER_PLAN_V1',
                source_stage=str(stage_root),host_skeleton=str(skeleton_root),
                create_host_skeleton_exclusively_before_container=True,placeholders=placeholders,
                apptainer_bind_arguments=[value for item in binds for value in ('--bind',item)],
                model_root=str(target/'base'),preset_file=str(target/'preset'/manifest['assets'][-1]['filename']),
                stage_root_inside='/zimage-sealed-assets',consumer_root_inside=str(target),
                mandatory_inside_verification='same tool --stage-root /zimage-sealed-assets --mounted-root /zimage-runtime-view',
                real_file_bind_execution_verified=False,source_link_or_copy_operations=0,
                independent_payload_SHA_verified=False,training_READY=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage-root',type=Path,default=STAGE)
    p.add_argument('--manifest',type=Path,default=REPO/MANIFEST)
    p.add_argument('--asset-id',type=int,action='append');p.add_argument('--mounted-root',type=Path)
    p.add_argument('--skeleton-root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    need(args.stage_root.is_absolute() and '..' not in args.stage_root.parts,'Explicit absolute stage root required')
    need(not args.output.resolve().is_relative_to(args.stage_root.resolve()),'No output writes inside sealed stage')
    if args.mounted_root is not None:
        need(not args.output.resolve().is_relative_to(args.mounted_root.resolve()),'No output writes inside consumer view')
    with args.manifest.open('rb') as f:raw=f.read(32769)
    need(len(raw)<=32768 and sha256(raw).hexdigest()==MANIFEST_SHA,'Immutable20 publisher manifest differs')
    manifest=json.loads(raw)
    result=verify_runtime(manifest,args.stage_root,args.asset_id,args.mounted_root)
    if args.skeleton_root is not None:
        result['consumer_plan']=file_bind_plan(manifest,result,args.stage_root,args.skeleton_root)
    body=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    with args.output.open('xb') as f:f.write(body)
    print(json.dumps({'state':result['state'],'planned':20,'verified':result['verified_sealed_assets'],
                      'payload_bytes_read':0,'output_SHA256':sha256(body).hexdigest(),'training_READY':False},sort_keys=True))


if __name__=='__main__':main()
