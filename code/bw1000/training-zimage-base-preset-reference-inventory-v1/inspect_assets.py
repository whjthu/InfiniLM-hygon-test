"""101 exact known asset slots; stat-only, no discovery or weight/body reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time


def need(ok,why):
    if not ok:raise ValueError(why)


def inspect(expected,bindings,check):
    groups=expected['groups']
    need(type(bindings) is dict and set(bindings)<=set(groups),'Unknown locator group')
    result={'schema':'ZIMAGE_TARGET_LISTED_ASSETS_STAT_TIER_V1','groups':{},
            'file_body_bytes_read':0,'tensor_values_read':0,'content_identity_qualified':False,
            'actual_model_loading':False,'actual_fusion':False,'actual_updates':0,'training_READY':False}
    planned=0;observed=0
    for group,spec in groups.items():
        rows={};value=bindings.get(group)
        need(value is None or type(value) is str and value!='','Locator must be an explicit absolute path or null')
        root=Path(value) if value is not None else None
        if root is not None:
            need(root.is_absolute() and '..' not in root.parts,'Relative/traversing asset locator forbidden')
        for name,pin in spec['files'].items():
            check();planned+=1
            relative=PurePosixPath(name)
            need(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,'Unsafe listed asset path')
            row={'expected_bytes':pin['bytes'],'expected_SHA256':pin['SHA256'],'body_SHA_verified':False,
                 'kind':'tensor_payload' if name.endswith(('.safetensors','.bin')) else 'reference_metadata_or_media'}
            if root is None:
                row['state']='TARGET_LOCATOR_UNBOUND'
            else:
                path=root/name if spec['locator_kind']=='directory' else root
                row['path']=str(path)
                try:
                    before=path.lstat()
                    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
                        row['state']='NONREGULAR_OR_LINK_NOT_FOLLOWED'
                    elif path.resolve(strict=True)!=path:
                        row['state']='ALIASED_PARENT_NOT_FOLLOWED'
                    else:
                        after=path.lstat()
                        a=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
                        b=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                        need(a==b,'Listed asset changed during stat observation')
                        row.update(observed_bytes=before.st_size,device=before.st_dev,inode=before.st_ino,
                                   mtime_ns=before.st_mtime_ns,
                                   state='SIZE_MATCH_CONTENT_UNVERIFIED' if before.st_size==pin['bytes'] else 'SIZE_MISMATCH')
                        observed+=1
                except FileNotFoundError:row['state']='ABSENT_AT_OBSERVATION'
                except PermissionError as error:row.update(state='PERMISSION_UNKNOWN',errno=error.errno)
            rows[name]=row
        counts={}
        for row in rows.values():counts[row['state']]=counts.get(row['state'],0)+1
        result['groups'][group]={'planned_files':len(rows),'locator_supplied':root is not None,
                                 'counts':counts,'files':rows}
    need(planned==101,'Frozen101 denominator differs')
    result.update(planned_files=planned,regular_stat_observed=observed,
                  state='STAT_OBSERVATIONS_ONLY_NO_CONTENT_OR_NATIVE_QUALIFICATION')
    return result


def main():
    start=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bindings',type=Path,required=True,help='JSON object of explicit group locators or null')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    package=Path(__file__).resolve().parent
    claim=json.loads((package/'CLAIM.json').read_bytes())
    need(claim.get('root_GO') is True and claim.get('READ_GO') is True
         and claim.get('GPU_GO') is False,'Root has not adopted this stat-only scope')
    need(args.bindings.suffix=='.json','Explicit locator JSON only')
    with args.bindings.open('rb') as f:raw=f.read(4097)
    need(len(raw)<=4096,'Locator JSON exceeds cap')
    bindings=json.loads(raw)
    expected=json.loads((package/'EXPECTED.json').read_bytes())
    def check():need(time.monotonic()-start<30,'Stat-only30s cooperative bound exceeded')
    value=inspect(expected,bindings,check)
    value.update(wall_seconds=time.monotonic()-start,expected_manifest_SHA256=hashlib.sha256((package/'EXPECTED.json').read_bytes()).hexdigest(),
                 actual_current_GPU_or_SDK_calls=0)
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    with args.output.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps({'planned_files':101,'regular_stat_observed':value['regular_stat_observed'],
                      'group_counts':{k:v['counts'] for k,v in value['groups'].items()},
                      'private_result':{'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()},
                      'file_body_bytes_read':0,'training_READY':False},sort_keys=True))


if __name__=='__main__':main()
