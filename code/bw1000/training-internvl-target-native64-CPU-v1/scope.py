"""One target-only CPU scope; pending permissions are a hard pre-import gate."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())
RUN=Path(P['RUN'])
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    raw=Path(path).read_bytes();require(len(raw)<=2*1024**2,'bounded metadata')
    return json.loads(raw)


def pin(path):
    raw=Path(path).read_bytes()
    return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def save(path,value,replace=False):
    raw=json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n'
    require(len(raw)<=1024**2,'output metadata cap')
    path=Path(path);target=path.with_suffix('.tmp') if replace else path
    with target.open('wb' if replace else 'xb') as stream:
        stream.write(raw)
    if replace:
        target.replace(path)


def authority_receipt(row,source_root):
    """The same published bytes under each host's own source root, never one cross-host absolute path."""
    require(type(row) is dict and type(row.get('bytes')) is int and 0<row['bytes']<=1024**2
            and re.fullmatch('[0-9a-f]{64}',row.get('SHA256','')),'QUALIFIED_AUTHORITY_PIN_REQUIRED')
    name=row.get('source_member');require(type(name) is str,'QUALIFIED_AUTHORITY_MEMBER_REQUIRED')
    part=PurePosixPath(name)
    require(not part.is_absolute() and '..' not in part.parts and '\\' not in name
            and part.as_posix()==name and len(part.parts)>=2 and part.parts[0]=='admission_receipts','BOUNDED_SOURCE_RELATIVE_AUTHORITY_MEMBER')
    origin=row.get('original_origin')
    require(type(origin) is dict and type(origin.get('host')) is str and bool(origin['host'])
            and type(origin.get('path')) is str and PurePosixPath(origin['path']).is_absolute()
            and '\\' not in origin['path'],'ORIGINAL_AUTHORITY_ORIGIN_REQUIRED')
    source_root=Path(source_root);receipt=source_root.joinpath(*part.parts)
    require(not any(path.is_symlink() for path in (receipt,*receipt.parents)),'AUTHORITY_MEMBER_LINKED')
    expected={key:row[key] for key in ('bytes','SHA256')}
    mf=read(source_root/'SOURCE_FILES.json')
    require(mf['files'].get(name)==expected,'AUTHORITY_MEMBER_NOT_SAME_PUBLISHED_SOURCE_MF')
    require(receipt.is_file() and receipt.stat().st_size==row['bytes'],'ACTUAL_AUTHORITY_MEMBER_BOUNDS')
    require(pin(receipt)==expected,'ACTUAL_AUTHORITY_MEMBER_SHA')
    return expected


def admission(claim,pre):
    # No import/probe/alternate loader may be used to discover these pending facts.
    require(claim['root_GO'] is True and claim['CPU_GO'] is True,'ROOT_GO_FALSE_SOURCE_ONLY')
    require(pre['root_GO'] is True and pre['pending_Torch_version_permission'] is False,
            'PENDING_TORCH_VERSION_NO_IMPORT_OR_LOADER')
    require(pre['sixshadow_original_state']=='LEGITIMATELY_RESOLVED_NO_REPLAY',
            'SIXSHADOW_UNKNOWN_NO_REQUERY_OR_ALTERNATE_IMPORT')
    require(pre['Hygon_wrapper_chain_qualified'] is True and type(pre['hygon_encode_wrapper_policy']) is dict,
            'HYGON_WRAPPER_QUALIFICATION_PENDING')
    for key in ('permission_resolution_evidence','hygon_provider_qualification','framework_source_receipt'):
        row=pre[key]
        require(type(row) is dict,'INDEPENDENT_PREREQUISITE_RECEIPT_PENDING_'+key)
        authority_receipt(row,D)
    root=pre['framework_root']
    require(type(root) is str and root.startswith('/public/home/') and '\\' not in root,'FRAMEWORK_ROOT_PENDING')
    policy=pre['hygon_encode_wrapper_policy']
    require(len(policy['chain'])==3 and policy.get('platform')=='HYGON_ACTUAL_INDEPENDENTLY_QUALIFIED',
            'A100_WRAPPER_POLICY_CANNOT_QUALIFY_HYGON')
    for row in policy['chain']:
        require(row['path'].startswith('/swift-runtime/site/') and '/torch/version' not in row['path'],
                'PENDING_VERSION_PATH_NOT_ALLOWED_AS_NEW_PROVIDER')
    require(type(pre.get('native_method_providers')) is dict,'HYGON_NATIVE_PROVIDERS_PENDING')
    return pre


def verify():
    mf=read(D/'SOURCE_FILES.json')
    for name,expected in mf['files'].items():
        path=D/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(),'source member path')
        require(pin(path)==expected,'source member bytes differ')
    return pin(D/'SOURCE_FILES.json')


def remaining(deadline,mono_end,reserve=0):
    seconds=min(deadline-time.time(),mono_end-time.monotonic())-reserve
    require(seconds>0,'ORIGINAL_900_EPOCH_EXHAUSTED')
    return seconds


def limits(seconds):
    require(type(seconds) in (float,int) and math.isfinite(seconds) and 0<seconds<=900,'original remaining seconds')
    slurm=60*(min(780,int(seconds)-90)//60)
    work=min(720,slurm-60)
    require(slurm>=120 and work>=60 and slurm+90<=seconds,'insufficient bounded whole-minute stage budget')
    return dict(Slurm_seconds=slurm,work_seconds=work,client_seconds=slurm+10,reserve_seconds=90)
