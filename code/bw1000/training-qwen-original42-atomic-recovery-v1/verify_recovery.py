"""Source/permission gates only; never import the original native/runtime code."""
import hashlib
import json
from pathlib import Path

H=lambda b:hashlib.sha256(b).hexdigest()


def need(ok,message):
    if not ok:
        raise ValueError(message)


def verify(root):
    root=Path(root);mf=json.loads((root/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        need(Path(name).name==name and not (root/name).is_symlink(),'SOURCE_MEMBER')
        raw=(root/name).read_bytes();need(len(raw)==pin['bytes'] and H(raw)==pin['SHA256'],'SOURCE_PIN')
    scope_raw=(root/'ACTION_SCOPE.json').read_bytes();scope=json.loads(scope_raw)
    claim=json.loads((root/'CLAIM.json').read_bytes());auth=json.loads((root/'AUTHORIZATION.json').read_bytes())
    need(claim['scope_SHA256']==auth['scope_SHA256']==H(scope_raw),'IMMUTABLE_ACTION_SCOPE')
    flags=tuple(claim[k] for k in ('root_GO','CPU_GO','GPU_GO','recovery_GO'))
    need(flags in ((False,False,False,False),(True,True,True,True)) and claim['performance_GO'] is False,'SOURCE_OR_ADOPTED_STATE')
    original=(root/'ORIGINAL_SOURCE_FILES.json').read_bytes()
    need(H(original)==scope['original_source_MF_SHA256']=='183476ed108621255be8d8bde114306340f035e6ddc13bbb69b902698bc3dda6','ORIGINAL17_SOURCE_MF')
    base=json.loads(original)['files'];target_raw=(root/'TARGET_SOURCE_FILES.json').read_bytes();target=json.loads(target_raw)['files']
    need(len(base)==len(target)==17 and set(base)==set(target),'SAME17_MEMBERS')
    need([n for n in base if base[n]!=target[n]]==['verify_package.py'],'ONLY_SINGLE_VERIFY_REPAIR')
    fixed=(root/'verify_package_fixed.py').read_bytes()
    need(target['verify_package.py']==dict(bytes=len(fixed),SHA256=H(fixed))
         and H(target_raw)==scope['target_source_MF_SHA256']==claim['target_source_MF_SHA256'],'REPAIRED_TARGET_MF')
    return mf,scope,claim,auth


def require_permission(scope,claim,auth,request_id,read_published):
    need(all(claim[k] is True for k in ('root_GO','CPU_GO','GPU_GO','recovery_GO')),'ROOT_REVIEW_NOT_ADOPTED')
    need(auth['explicit_user_exception_received'] is True and auth['approved_new_request_id']==request_id
         and request_id!=scope['original_request'] and auth['original694_authorized'] is False,
         'EXPLICIT_ORIGINAL42_USER_EXCEPTION_REQUIRED')
    pin=auth['published_user_record'];path=pin['path']
    need(isinstance(path,str) and path.startswith(('evidence/bw1000/','results/bw1000/'))
         and '..' not in Path(path).parts and '\\' not in path,'PUBLISHED_USER_AUTHORITY_PATH')
    need(type(pin['bytes'])is int and 0<pin['bytes']<=16384,'SMALL_USER_AUTHORITY_PIN')
    raw=read_published(path)
    need(len(raw)==pin['bytes'] and H(raw)==pin['SHA256'],'PUBLISHED_USER_AUTHORITY_BYTES')
    record=json.loads(raw)
    need(record['user_authorized'] is True and record['exception_for_original_UNKNOWN'] is True
         and record['accepts_residual_original_execution_UNKNOWN'] is True
         and record['original_request']==scope['original_request'] and record['scope_SHA256']==claim['scope_SHA256']
         and record['target_source_MF_SHA256']==scope['target_source_MF_SHA256']
         and record['approved_new_request_id']==request_id and record['maximum_new_invocations']==1
         and record['original694_authorized'] is False,'EXACT_USER_EXCEPTION_SCOPE')
    return dict(path=path,bytes=len(raw),SHA256=H(raw))
