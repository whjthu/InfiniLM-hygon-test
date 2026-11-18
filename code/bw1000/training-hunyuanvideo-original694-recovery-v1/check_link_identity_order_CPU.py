"""One tiny .bin regression: partial is gone before the unique final read/stamp."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import reuse,atomic_files

def main():
    reuse.DEADLINE=time.time()+20;reuse.MONO_END=time.monotonic()+20
    shims=[]
    for name in ('O_NOFOLLOW','O_NONBLOCK'):
        if not hasattr(os,name):setattr(os,name,0);shims.append(name+'=0_TEST_ONLY')
    if not hasattr(os,'getuid'):os.getuid=lambda:0;shims.append('getuid=Windows_st_uid0_TEST_ONLY')
    if os.name=='nt':
        def stamp(info):return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]
        reuse.stable=stamp;atomic_files.stable=stamp;shims.append('Windows_temp_ctime_adapter_ONLY_Linux5fields_unchanged')
    events=[];reads=[];original_existing=atomic_files.existing
    with tempfile.TemporaryDirectory(prefix='T09-hardlink-order-') as folder:
        root=Path(folder);request='5'*32;partial=root/('.new.bin.'+request+'.partial')
        def sync(path):events.append(dict(operation='directory_sync_hook',partial_exists=partial.exists()))
        def checked_existing(path,name,pin,maximum):
            assert not partial.exists(),'FINAL_READ_BEFORE_OWN_PARTIAL_UNLINK'
            reads.append(name);events.append(dict(operation='single_final_read',partial_exists=False))
            return original_existing(path,name,pin,maximum)
        atomic_files.sync_directory=sync;atomic_files.existing=checked_existing
        data=b'no-tensor-test';pin=dict(bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())
        result=atomic_files.install(root,'new.bin',data,pin,request,64)
        assert reads==['new.bin'] and not partial.exists() and (root/'new.bin').stat().st_nlink==1
        assert result['file_identity']==reuse.stable((root/'new.bin').stat())
        assert events[0]==dict(operation='directory_sync_hook',partial_exists=False)
        assert events[1]==dict(operation='single_final_read',partial_exists=False)
    value=dict(passed=True,new_regression_boundaries=1,final_payload_reads=1,events=events,
        final_identity_after_partial_unlink=True,production_Linux5field_relaxed=False,
        original_tests_repeated=False,portability_shims=shims,Linux_NFS_durability_qualified=False,
        PT_reads=0,network_calls=0,requests_created=0)
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    out=Path(__file__).resolve().parents[3]/'local-only/training-hunyuanvideo-original694-recovery-v1-localchecks/link-identity-order-first.json'
    with out.open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(passed=True,path=str(out),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),final_payload_reads=1)))

if __name__=='__main__':main()
