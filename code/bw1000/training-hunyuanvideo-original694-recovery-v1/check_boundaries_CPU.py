"""Temporary tiny .bin data only; no .pt, network, subprocess, Git or permission grant."""
import copy,hashlib,json,os,tempfile,time
from pathlib import Path
import reuse,atomic_files
from reuse import pack,unpack
from permission_gate import validate
from relay_once import transport_state

def main():
    passed=[];rejected=[];shims=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,EOFError) as error:rejected.append(dict(name=name,reason=str(error)))
        else:raise AssertionError('INVALID_ACCEPTED:'+name)
    reuse.DEADLINE=time.time()+30;reuse.MONO_END=time.monotonic()+30
    # The runtime is Linux. This Windows test only tests byte/no-replace decisions,
    # and expressly does not qualify Linux O_NOFOLLOW, NFS or directory-fsync behavior.
    for name in ('O_NOFOLLOW','O_NONBLOCK'):
        if not hasattr(os,name):setattr(os,name,0);shims.append(name+'=0_TEST_ONLY')
    if not hasattr(os,'getuid'):
        os.getuid=lambda:0;shims.append('getuid=Windows_st_uid0_TEST_ONLY')
    atomic_files.sync_directory=lambda root:None
    shims.append('directory_fsync_NOT_TESTED_on_Windows')
    if os.name=='nt':
        # Windows fstat ctime vs path.stat ctime use different semantics. This is only
        # a temp-fixture adapter; the production Linux five-field function is untouched.
        def fixture_stamp(info):return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]
        reuse.stable=fixture_stamp;atomic_files.stable=fixture_stamp
        shims.append('Windows_temp_fixture_ctime_omitted_TEST_ONLY_Linux5field_unchanged')
    data=b'abc123';pin=dict(bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())
    with tempfile.TemporaryDirectory(prefix='T09-opaque-boundary-') as folder:
        root=Path(folder)
        new=atomic_files.install(root,'new.bin',data,pin,'1'*32,64)
        assert new['state']=='NEW_ATOMIC_NOREPLACE' and (root/'new.bin').read_bytes()==data
        passed.append('same_filesystem_exclusive_install_tiny_bin')
        prior=(root/'new.bin').stat().st_ino
        found=atomic_files.existing(root,'new.bin',pin,64)
        assert found['state']=='EXISTING_CORRECT_REUSED'
        again=atomic_files.install(root,'new.bin',data,pin,'2'*32,64)
        assert again['state']=='CONCURRENT_CORRECT_REUSED' and (root/'new.bin').stat().st_ino==prior
        passed.append('correct_existing_and_concurrent_correct_reused_no_inode_replace')
        (root/'bad.bin').write_bytes(b'xxxxxx')
        reject('different_existing_same_size_no_overwrite',lambda:atomic_files.install(root,'bad.bin',data,pin,'3'*32,64))
        assert (root/'bad.bin').read_bytes()==b'xxxxxx' and not list(root.glob('*.partial'))
        reject('path_traversal',lambda:atomic_files.install(root,'../escape.bin',data,pin,'4'*32,64))
    plan=dict(max_file_bytes=64,max_index_bytes=4096,max_decompressed_bytes=8192,max_compressed_bytes=8192)
    context=dict(request_id='synthetic-only')
    blob=pack({'a.bin':data},plan,context);assert unpack(blob,plan,context)=={'a.bin':data}
    passed.append('qualified_pack_unpack_tiny_opaque_bytes')
    reject('truncated_gzip',lambda:unpack(blob[:-2],plan,context))
    base=Path(__file__).resolve().parent
    actual_plan=json.loads((base/'PLAN.json').read_bytes());claim=json.loads((base/'CLAIM.json').read_bytes())
    reject('allGO_false_never_permission',lambda:validate(claim,{},actual_plan))
    dummy=copy.deepcopy(claim)
    dummy.update(root_GO=True,CPU_GO=True,transfer_GO=True,explicit_user_exception_received=True,
        exception_record_path='synthetic-only.json',exception_record_SHA256='0'*64)
    approval=dict(schema='T09_694_EXPLICIT_USER_EXCEPTION_V1',user_granted=True,allow_atomic_recovery_despite_unknown=True,
        original_request_id=actual_plan['original_request_id'],original_state='UNKNOWN',target=actual_plan['target'],
        members=actual_plan['members'],max_invocations=1,epoch_seconds=300,no_overwrite=True,no_target_rename=True)
    assert validate(dummy,approval,actual_plan) is True
    passed.append('synthetic_permission_schema_ONLY_not_actual_permission')
    bad=copy.deepcopy(approval);bad['target']+='/foreign'
    reject('exception_cannot_change_target',lambda:validate(dummy,bad,actual_plan))
    bad=copy.deepcopy(approval);bad['original_state']='NOT_EXECUTED'
    reject('exception_cannot_wash_oldUNKNOWN',lambda:validate(dummy,bad,actual_plan))
    assert transport_state(0,False,[],False)=='RETURNED'
    assert transport_state(0,True,[],False)=='UNKNOWN'
    assert transport_state(255,False,[],False)=='UNKNOWN'
    passed.append('forced_kill_persists_UNKNOWN_after_zero_poll')
    out=base.parents[2]/'local-only/training-hunyuanvideo-original694-recovery-v1-localchecks'
    out.mkdir(exist_ok=False)
    value=dict(passed=True,normal_count=len(passed),rejected_count=len(rejected),normal=passed,rejected=rejected,
        portability_shims=shims,Linux_NFS_durability_qualified=False,actual_permission_granted=False,
        original_PT_reads=0,network_calls=0,requests_created=0,subprocess_calls=0,Git_calls=0)
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();(out/'first-boundaries.json').write_bytes(raw)
    print(json.dumps(dict(passed=True,normal=len(passed),rejected=len(rejected),path=str(out/'first-boundaries.json'),
        bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())))

if __name__=='__main__':main()
