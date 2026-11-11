"""Exercise the real extraction parser using tiny CPU bytes, never model artifacts."""
import copy,gzip,hashlib,io,json,tarfile,tempfile
from pathlib import Path
from build_archive import safe_name
from receive_archive import extract_pinned_archive

def make_archive(path, plan, change=None):
    raw=json.dumps(plan).encode()
    with tarfile.open(path,'w:gz',format=tarfile.USTAR_FORMAT) as tar:
        rows=[('assets/'+r['name'],b'abc') for r in plan['files']]+[('ASSET_PLAN.json',raw)]
        for i,(name,data) in enumerate(rows):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            if change:info,data=change(i,info,data)
            tar.addfile(info,io.BytesIO(data))
    return raw

def run(plan,change=None):
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);archive=root/'pack.tar.gz';raw=make_archive(archive,plan,change)
        out=root/'out';out.mkdir();done=[]
        extract_pinned_archive(archive,out,plan,raw,lambda:None,done)
        assert len(done)==76 and len(list((out/'assets').rglob('*.bin')))==76

plan={'files':[dict(name='rank-%02d.bin'%i,bytes=3,SHA256=hashlib.sha256(b'abc').hexdigest()) for i in range(76)]}
run(plan);rejected=[]
def reject(name,func):
    try:func()
    except (AssertionError,KeyError,tarfile.TarError,EOFError):rejected.append(name);return
    raise AssertionError('bad archive accepted: '+name)

def mutate(field,value):
    def change(i,info,data):
        if i==1:setattr(info,field,value)
        return info,data
    return change

for name,change in [
    ('parent_escape',mutate('name','../escape')),
    ('absolute_path',mutate('name','/outside')),
    ('duplicate_member',mutate('name','assets/rank-00.bin')),
    ('foreign_member',mutate('name','assets/not-in-plan.bin')),
    ('symlink',mutate('type',tarfile.SYMTYPE)),
    ('hardlink',mutate('type',tarfile.LNKTYPE)),
    ('wrong_mode',mutate('mode',0o777)),
    ('wrong_uid',mutate('uid',1000)),
    ('wrong_mtime',mutate('mtime',1)),
]:reject(name,lambda change=change:run(plan,change))

def bad_bytes(i,info,data):return (info,b'xyz') if i==1 else (info,data)
reject('payload_SHA',lambda:run(plan,bad_bytes))
def bad_manifest(i,info,data):return (info,b' '*len(data)) if info.name=='ASSET_PLAN.json' else (info,data)
reject('plan_bytes',lambda:run(plan,bad_manifest))
for name in ('../x','/x','a/../../x','a\\x','a//x',''):
    reject('path:'+name,lambda name=name:safe_name(name))
assert len(rejected)==17
print('PASS: exact76 CPU payloads plus plan; 17 malformed path/member/metadata/hash rejections using actual receiver extraction code. No GPU or original artifact read.')
