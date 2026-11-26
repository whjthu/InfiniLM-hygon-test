"""Authenticated target entry for one T09 source install, owner start, or terminal read."""
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import zlib


R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='t09-backward67-prefix-b328350dc06b'
REQUEST='b328350dc06b42d7805ae1fbae00eb3b'
STAGE_REQUEST='8d65f659964a4e9d8ab1b25683c4e202'
READ_REQUEST='ddae8c7d31d148a4bf2efee52043bf4a'
RUN=R/'results'/LABEL
SRC=RUN/'src'
PARENT,NODE='1103606','f13r3n13'
PYTHON='/usr/local/python3.12/bin/python3'
MF_SHA='385aba78b54293cedf9925532397a42266a1fa78e678e8826f4dad5a87c75667'
CLAIM_SHA='3f9813bd8ead9c592d8a3fd1ecf0c3f9daed6bbfdf318c7082aa3717c46ec823'
ELF_SCOPE=R/'incoming/t09-miopen-single-bind-c6025678606b/work-once'
ELF=ELF_SCOPE/'provider/libMIOpen.so.1.0'
ELF_PIN=dict(bytes=18705784,SHA256='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba')
ELF_READBACK='24b1cad945b6a04f8fc088f2c504b24a681bd6039499910be93a7e1a5a3d2f3c'
FAILED_3616='1ad8f0ed1b88d517ab301f9281da5d0c42c63e6afd54791eac7a66f352d2a336'


def need(ok,code):
    if not ok:raise ValueError(code)


def encode(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())
def identity(info):return [info.st_dev,info.st_ino,info.st_size,info.st_mode,info.st_uid,info.st_mtime_ns,info.st_ctime_ns]


def read(path,cap):
    path=Path(path)
    need(path.resolve(strict=True)==path,'EXACT_CANONICAL_PROJECT_MEMBER')
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'EXACT_BOUNDED_ORIGINAL_EXTENT')
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        need(identity(before)==identity(os.fstat(fd)),'SAME_OPEN_DESCRIPTOR')
        raw=stream.read(cap+1)
        need(identity(before)==identity(os.fstat(fd))==identity(path.lstat()) and len(raw)==before.st_size,
             'STABLE_SAME_DESCRIPTOR_BEFORE_AFTER_READ')
    return raw,before


def dir0700(path):
    need(path.resolve(strict=True)==path and path.is_dir() and path.stat().st_uid==os.getuid()
         and stat.S_IMODE(path.stat().st_mode)==0o700,'EXACT_OWNED_REQUEST_DIRECTORY_0700')


def syncdir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


def exact_elf():
    old_raw,_=read(ELF_SCOPE/'compute-host-copy-readback.json',8192)
    old=json.loads(old_raw)
    need(pin(old_raw)['SHA256']==ELF_READBACK and old['pin']==ELF_PIN and old['owner_mode_octal']=='0400',
         'EXACT_3600_ORIGINAL_COMPUTE_READBACK')
    raw,info=read(ELF,ELF_PIN['bytes'])
    need(pin(raw)==ELF_PIN and stat.S_IMODE(info.st_mode)==0o400 and info.st_uid==os.getuid()
         and info.st_nlink==1 and info.st_ino==old['dev_inode'][1],'CURRENT_EXACT_3600_SEALED_ELF')
    return dict(LIB=ELF_PIN,prior_compute_readback_SHA256=ELF_READBACK,mode='0400',
                owner_uid_matches=True,original_compute_inode_matches=True,host_device_rebase_allowed=True)


def check_members(mf,bodies):
    need(pin(mf)['SHA256']==MF_SHA and len(mf)==4453,'EXACT_PUBLISHED_V3_MANIFEST')
    manifest=json.loads(mf)
    members=manifest['files']
    need(manifest['label']==LABEL and manifest['request']==REQUEST and len(members)==33 and set(members)==set(bodies),
         'EXACT_33_STAGED_MEMBERS')
    for name,value in members.items():
        need(re.fullmatch('[A-Za-z_0-9]+[.](?:json|py)',name) is not None and 0<len(bodies[name])<=65536
             and pin(bodies[name])==value,'EXACT_MEMBER_TYPE_BYTES_AND_SHA')
    need(pin(bodies['CLAIM.json'])['SHA256']==CLAIM_SHA,'EXACT_PUBLISHED_T09_OPERATIVE_CLAIM')
    return manifest


def check_install(expected=None):
    dir0700(RUN)
    dir0700(SRC)
    mf,_=read(SRC/'SOURCE_FILES.json',16384)
    spec=json.loads(mf)['files']
    need(set(spec)|{'SOURCE_FILES.json'}=={name.name for name in SRC.iterdir()},'EXACT_DELIVERED_SOURCE_NAMES')
    bodies={name:read(SRC/name,65536)[0] for name in spec}
    check_members(mf,bodies)
    raw,_=read(RUN/'installed-source.json',16384)
    staged=json.loads(raw)
    need(staged['stage_request']==STAGE_REQUEST and staged['scope']==REQUEST and staged['source_manifest']==pin(mf)
         and staged['target_R_and_src_mode']=='0700' and staged['original_3600_elf']['LIB']==ELF_PIN,
         'EXACT_PREVIOUS_SUCCESSFUL_SOURCE_INSTALL')
    if expected is not None:need(pin(raw)['SHA256']==expected,'INDEPENDENT_PREVIOUS_INSTALL_RECEIPT_SHA')
    return staged,pin(raw)


def stage(a):
    need(a['request']==STAGE_REQUEST and re.fullmatch('[0-9a-f]{40}',a['publication']) is not None,
         'EXACT_NEW_STAGE_AND_PUBLICATION')
    d=zlib.decompressobj()
    data=d.decompress(b64decode(a['packet'],validate=True),380001)
    need(d.eof and not d.unconsumed_tail and not d.unused_data and 0<len(data)<=380000,'BOUNDED_EXACT_SOURCE_PACKET')
    archive=json.loads(data)
    need(set(archive)=={'manifest','files'},'SOURCE_ONLY_STAGE_PACKET')
    raw=b64decode(archive['manifest'],validate=True)
    bodies={name:b64decode(body,validate=True) for name,body in archive['files'].items()}
    check_members(raw,bodies)
    elf=exact_elf()
    lock=R/'incoming'/(LABEL+'-stage-'+STAGE_REQUEST)
    need(not os.path.lexists(lock) and not os.path.lexists(RUN),'NEW_UNUSED_INSTALL_AND_NO_USER_WORKSPACE')
    lock.mkdir(mode=0o700)
    write(lock/'intent.json',encode(dict(request=STAGE_REQUEST,scope=REQUEST,source_manifest=pin(raw),publication=a['publication'])))
    RUN.mkdir(mode=0o700)
    SRC.mkdir(mode=0o700)
    dir0700(lock)
    dir0700(RUN)
    dir0700(SRC)
    for name,body in bodies.items():write(SRC/name,body)
    write(SRC/'SOURCE_FILES.json',raw)
    syncdir(SRC)
    need(set(bodies)|{'SOURCE_FILES.json'}=={path.name for path in SRC.iterdir()}
         and all(read(SRC/name,65536)[0]==body for name,body in bodies.items())
         and read(SRC/'SOURCE_FILES.json',16384)[0]==raw,'INDEPENDENT_INSTALL_REREAD_ALL_33')
    out=dict(request=STAGE_REQUEST,stage_request=STAGE_REQUEST,scope=REQUEST,publication=a['publication'],
             source_manifest=pin(raw),original_3600_elf=elf,target_R_and_src_mode='0700',
             independently_reloaded_members=33,actual_GPU=0,controller_started=False)
    installed=write(RUN/'installed-source.json',encode(out))
    syncdir(RUN)
    write(lock/'completed.json',encode(dict(out,installed=installed)))
    return dict(out,installed=installed)


def terminal(a):
    need(a['request']==READ_REQUEST and isinstance(a['epoch'],(float,int)),'DISTINCT_TERMINAL_READ_REQUEST')
    staged,installed=check_install(a['installed_SHA256'])
    out_raw,_=read(RUN/'outcome.json',256<<10)
    out=json.loads(out_raw)
    need(out['request_id']==REQUEST and out['epoch']==a['epoch'] and out['source_manifest_SHA256']==MF_SHA
         and pin(out_raw)['SHA256']==a['outcome_SHA256'],'EXACT_ORIGINAL_TERMINAL_OUTCOME_BEFORE_INDEPENDENT_READ')
    pf_raw,_=read(RUN/'preflight.json',16384)
    pf=pin(pf_raw)['SHA256']
    need(pf==a['preflight_SHA256'],'SAME_ORIGINAL_PREFLIGHT_FOR_READ')
    output=RUN/'independent-terminal.json'
    need(not os.path.lexists(output),'ORIGINAL_TERMINAL_READER_ALREADY_ATTEMPTED')
    lock=R/'incoming'/(LABEL+'-terminal-'+READ_REQUEST)
    lock.mkdir(mode=0o700)
    dir0700(lock)
    write(lock/'intent.json',encode(dict(request=READ_REQUEST,scope=REQUEST,source_manifest=staged['source_manifest'],
               original_outcome=pin(out_raw),preflight=pin(pf_raw),independent_output_name=output.name)))
    syncdir(lock)
    entry=('import runpy,sys;sys.path.insert(0,'+repr(str(SRC))+');sys.argv='+repr([
        str(SRC/'read_terminal.py'),'--epoch',str(a['epoch']),'--source-manifest-sha256',MF_SHA,
        '--preflight-sha256',pf,'--output',str(output)])+';runpy.run_path(sys.argv[0],run_name="__main__")')
    done=subprocess.run([PYTHON,'-I','-S','-B','-u','-c',entry],stdin=subprocess.DEVNULL,
                        capture_output=True,timeout=30)
    need(len(done.stdout)<=16384 and len(done.stderr)<=65536,'BOUNDED_INDEPENDENT_READER_STDIO')
    value_raw,_=read(output,16384)
    value=json.loads(value_raw)
    need(value['request_id']==REQUEST and value['planned_updates']==7
         and value['completed_updates']+value['missing_updates']==7,'EXACT_INDEPENDENT_DENOMINATOR')
    return dict(request=READ_REQUEST,scope=REQUEST,source_manifest=staged['source_manifest'],install=installed,
                original_outcome=pin(out_raw),independent=pin(value_raw),independent_rc=done.returncode,
                passed=value['passed'],independent_release=value.get('independent_release',False),
                planned_updates=7,completed_updates=value['completed_updates'],missing_updates=value['missing_updates'],
                actual_new_GPU_steps=0,raw_log_printed=False)



CPU_LABEL='t09-prefix7-physical-08f92cd20b51'
CPU_REQUEST='08f92cd20b514b8492b4925c67dee4e8'
PHASES=('numeric_cgroup', 'live_step_resource', 'passive_DRM_KFD_same_UID_FD', 'compute_host_model_stat', 'original_apptainer_launch', 'container_weight_stat_and_effective_ro')
PREFLIGHT=b'{"cache": {"SHA256": "9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b", "bytes": 3521630, "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-original-inputs-v1/sample.pt"}, "config": {"SHA256": "689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae", "bytes": 968, "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-original-inputs-v1/config.json"}, "deadline": 1795633518.8272123, "epoch": 1795632078.8272123, "framework_manifest_SHA256": "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41", "model": {"SHA256": "71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86", "bytes": 33306632192, "file_identity": [49, 11025132338, 33306632192, 1794715911000000000, 1794715912000000000], "full_SHA_receipt": {"SHA256": "8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4", "bytes": 932, "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/T09-native25-original-transformer-rebind-fullsha-CPU-v1/normalized-independent-terminal.json"}, "host_file_identity": [47, 11025132338, 33306632192, 1794715911000000000, 1794715912000000000], "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model.safetensors", "verification": "EXECUTION_MOUNT_FULL_SHA256"}, "native_source_manifest": {"SHA256": "26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7", "bytes": 7689, "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2/src/NATIVE_BINDING.json"}, "node": "f13r3n13", "parent": "1103606", "passed": true, "request_id": "a0b6de47bf31408eb41796e8d54602d8", "root_GO": true, "run": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2", "runtime_manifest": {"SHA256": "56f65e63b28f8ec30c7dc1efb3f6dc1df291d4a5d981f91d681f2fad41a03b6a", "bytes": 2124, "path": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t06-sdxl-runtime-v1/receipt.json"}, "state": "T09_CACHED_NATIVE_TARGET_PREFLIGHT_PASS"}\n'
GUARDED_WORKER='import os,json\nif (os.getuid()!=154459 or any(os.environ.get(k) for k in (\'MASTER_ADDR\',\'MASTER_PORT\',\'TORCHELASTIC_RUN_ID\',\'RDZV_ENDPOINT\'))\n        or os.environ.get(\'WORLD_SIZE\',\'1\')!=\'1\'):\n    print(json.dumps(dict(passed=False,failed_phase=\'exact_compute_UID_and_no_rendezvous\')),flush=True)\n    raise SystemExit(2)\nGUEST=\'\\nimport hashlib,json,os,sys\\nfrom pathlib import Path\\ntry:\\n    path=Path(sys.argv[1])\\n    if path.resolve(strict=True)!=path:\\n        raise ValueError(\\\'CONTAINER_MODEL_PATH_NOT_CANONICAL\\\')\\n    stamp=path.lstat()\\n    result=dict(passed=True,mount_readonly=bool(os.statvfs(path).f_flag & os.ST_RDONLY),\\n         file_identity=[stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns],weight_bytes_hashed=0)\\nexcept BaseException as error:\\n    name=str(error)\\n    result=dict(passed=False,error_type=type(error).__name__,errno=getattr(error,\\\'errno\\\',None),\\n                error_code=name if name==\\\'CONTAINER_MODEL_PATH_NOT_CANONICAL\\\' else None,\\n                message_SHA256=hashlib.sha256(name.encode(\\\'utf8\\\',\\\'replace\\\')).hexdigest())\\nprint("T09_MOUNT_RESULT_V3 "+json.dumps(result,sort_keys=True),flush=True)\\nraise SystemExit(0 if result[\\\'passed\\\'] else 3)\\n\'\nPREFIX=b\'T09_MOUNT_RESULT_V3 \'\ndef projection(error):\n    import traceback\n    import re\n    names={\'placement.py\',\'snapshot.py\',\'<string>\',\'subprocess.py\'}\n    raw=str(error)\n    return dict(error_type=type(error).__name__,errno=getattr(error,\'errno\',None),\n        error_code=raw if re.fullmatch(\'[A-Z0-9_]{1,96}\',raw) else None,\n        message_SHA256=sha256(raw.encode(\'utf8\',\'replace\')).hexdigest(),\n        frames=[dict(file=Path(row.filename).name,line=row.lineno,\n                     function=row.name if re.fullmatch(\'[A-Za-z_][A-Za-z0-9_]*|<module>\',row.name) else \'OTHER\')\n                for row in traceback.extract_tb(error.__traceback__)[-6:] if Path(row.filename).name in names])\n\ndef extract_error(raw):\n    text=raw.decode(\'utf8\',\'replace\')\n    codes=(\'T09_NUMERIC_STEP\',\'T09_EXACT_NUMERIC_STEP_CGROUP\',\'T09_EXACT_COMPUTE_NODE\',\n           \'T09_ONE_EXACT_LIVE_STEP_RECORD\',\'T09_NATIVE_STEP_RESOURCE_PROOF\')\n    classes=(\'AssertionError\',\'PermissionError\',\'CalledProcessError\',\'TimeoutExpired\',\'JSONDecodeError\',\n             \'RuntimeError\',\'FileNotFoundError\',\'ValueError\')\n    return dict(codes=[code for code in codes if code in text],\n                exception_classes=[name for name in classes if re.search(r\'(?m)^[\\w.]*\'+name+r\'(?:\\s*[:\\r\\n]|$)\',text)],\n                traceback_frame_lines=[int(n) for n in re.findall(r\'(?m)^  File "[^"]{1,300}", line (\\d{1,6}), in .+$\',text)[:6]],\n                raw_paths_or_messages_printed=False)\n\ndef stdout_class(raw):\n    rows=raw.splitlines()\n    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest(),prefix=\'T09_MOUNT_RESULT_V3\',\n        marker_lines=sum(row.startswith(PREFIX) for row in rows),\n        other_nonblank_lines=sum(bool(row.strip()) and not row.startswith(PREFIX) for row in rows),\n        blank_lines=sum(not row.strip() for row in rows),raw_printed=False)\n\ndef marked_result(raw,projection):\n    def require(flag,label):\n        if not flag:\n            raise ValueError(label)\n    require(projection[\'marker_lines\']==1 and projection[\'other_nonblank_lines\']==0,\n            \'EXPECTED_ONE_CONTROLLED_CONTAINER_STDOUT_MARKER_ONLY\')\n    found,=[row[len(PREFIX):] for row in raw.splitlines() if row.startswith(PREFIX)]\n    guest=json.loads(found)\n    require(isinstance(guest,dict) and type(guest.get(\'passed\')) is bool,\'EXPECTED_FRAMED_JSON_OBJECT\')\n    if guest[\'passed\']:\n        require(set(guest)=={\'passed\',\'mount_readonly\',\'file_identity\',\'weight_bytes_hashed\'}\n                and type(guest[\'mount_readonly\']) is bool and type(guest[\'weight_bytes_hashed\']) is int\n                and type(guest[\'file_identity\']) is list and len(guest[\'file_identity\'])==5\n                and all(type(x) is int and x>=0 for x in guest[\'file_identity\']),\n                \'EXPECTED_FRAMED_MOUNT_SUCCESS_SCHEMA\')\n    else:\n        require(set(guest)=={\'passed\',\'error_type\',\'errno\',\'error_code\',\'message_SHA256\'}\n                and isinstance(guest[\'error_type\'],str) and isinstance(guest[\'message_SHA256\'],str),\n                \'EXPECTED_FRAMED_MOUNT_FAILURE_SCHEMA\')\n    return guest\n\n\nimport json,os,re,stat,subprocess,sys,time\nfrom pathlib import Path\nfrom hashlib import sha256\n\ndef need(ok,code):\n    if not ok:raise ValueError(code)\n\ndef worker():\n    source,label,work=sys.argv[1:]\n    work=Path(work)\n    sys.path.insert(0,source)\n    from placement import authentic_step,numeric_cgroup\n    from contract import ROOT,WEIGHT,COMPUTE_MODEL_STAMP,stamp\n    from snapshot import snapshot\n    results={}\n    report=dict(passed=False,phase_results=results,zero_GPU=True)\n    phase=\'numeric_cgroup\'\n    try:\n        handle=numeric_cgroup()\n        report[\'step\']=handle\n        need(not os.environ.get(\'SLURM_STEP_GPUS\'),\'REJECT_UNEXPECTED_STEP_GPU_SELECTION\')\n        results[phase]=True\n        phase=\'live_step_resource\'\n        need(authentic_step(\'pre\',label)==handle,\'EXACT_LIVE_ZERO_GRES_STEP\')\n        results[phase]=True\n        phase=\'passive_DRM_KFD_same_UID_FD\'\n        doc=snapshot(False)\n        report[\'actual_physical\']=doc\n        need(doc[\'step\']==handle and doc[\'node\']==\'f13r3n13\'\n            and len(doc[\'pci\'])==len(set(doc[\'pci\']))==8 and doc[\'kfd_holder_count\']==0,\n            \'ACTUAL_OLD_NODE_NEEDS_COMPLETE_ZERO_HOLDER_INVENTORY\')\n        results[phase]=True\n        phase=\'compute_host_model_stat\'\n        weight=Path(WEIGHT)\n        compute=stamp(weight.lstat())\n        report[\'compute_host_model_identity\']=compute\n        need(weight.resolve(strict=True)==weight and compute==COMPUTE_MODEL_STAMP,\n            \'COMPUTE_HOST_MODEL_APERTURE_DIFFERS\')\n        results[phase]=True\n        phase=\'original_apptainer_launch\'\n        os.chdir(\'/\')\n        setup=\'set -euo pipefail; source /opt/dtk/env.sh >&2; unset PYTHONPYCACHEPREFIX; export ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; exec python3 -I -S -B -c "$@"\'\n        argv=[\'/public/software/apps/apptainer/1.3.4/bin/apptainer\',\'exec\',\'--rocm\',\'--no-mount\',\'home,cwd\',\n              \'--pwd\',\'/\',\'--bind\',f\'{ROOT}:{ROOT}:ro\',\'--bind\',\'/opt/hyhal:/opt/hyhal:ro\',\n              str(ROOT/\'containers/dtk26-sandbox-r1\'),\'/bin/bash\',\'--noprofile\',\'--norc\',\'-c\',\n              setup,\'t09-n13-typed-passive-mount\',GUEST,str(WEIGHT)]\n        result=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=12)\n        need(len(result.stdout)<=8192 and len(result.stderr)<=32768,\'EXACT_BOUNDED_CONTAINER_CAPTURE\')\n        need(work.resolve(strict=True)==work and stat.S_IMODE(work.stat().st_mode)==0o700\n             and work.stat().st_uid==os.getuid(),\'OWN_UNIQUE_PRIVATE_CAPTURE_DIRECTORY\')\n        with (work/\'container.stderr.private\').open(\'xb\') as stream:\n            stream.write(result.stderr);stream.flush();os.fsync(stream.fileno())\n        report[\'apptainer\']=dict(rc=result.returncode,stderr_bytes=len(result.stderr),\n            stderr_sha256=sha256(result.stderr).hexdigest(),known_stderr=extract_error(result.stderr))\n        with (work/\'container.stdout.private\').open(\'xb\') as stream:\n            stream.write(result.stdout);stream.flush();os.fsync(stream.fileno())\n        report[\'container_stdout\']=stdout_class(result.stdout)\n        guest=marked_result(result.stdout,report[\'container_stdout\'])\n        results[phase]=True\n        phase=\'container_weight_stat_and_effective_ro\'\n        report[\'container_observation\']=guest\n        need(result.returncode==0 and guest[\'passed\'] is True and guest[\'mount_readonly\'] is True\n             and guest[\'file_identity\']==compute and guest[\'weight_bytes_hashed\']==0,\n             \'ACTUAL_IMAGE_WEIGHT_MOUNT_OR_METADATA_DIFFERS\')\n        results[phase]=True\n        report.update(passed=True,observed_epoch=time.time())\n    except BaseException as error:\n        report.update(failed_phase=phase,worker_failure=projection(error),observed_epoch=time.time())\n        results[phase]=False\n    print(json.dumps(report,sort_keys=True),flush=True)\n    return 0 if report[\'passed\'] else 2\nraise SystemExit(worker())\n'
ORIGINAL_3637=['1103606.3637', 't09-untraced-n13-framed-cpu-4c6e3b096b9d', 'COMPLETED', '0:0', '1', 'f13r3n13', 'cpu=1,mem=1G,node=1', '2026-11-25T22:52:29', '2026-11-25T22:52:30']

"""One approved HOST physical admission and detached prefix7 controller."""

# The renderer prepends the exact staged-source helpers, bounded worker and metadata template.

WHOLE=1440
EXPECTED_UID=154459
PRIVATE_NAMES={'preflight.json','placement-authority.json','physical-route.json'}
CPU_SHA='c9f071765f0085e141d6e7b7fee5fa6252a3f72d42c36b485d196bbb2b3d6fc2'
GPU_ARCHIVE='34d113536c88f7ebc9fa06c3218324f4cb5f31e89cc382077c6e032f01fc46e4'
PORT_VARS=('MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID','RDZV_ENDPOINT')


def no_ports(env):
    need(not any(env.get(k) for k in PORT_VARS) and env.get('WORLD_SIZE','1')=='1',
         'ORIGINAL_SINGLE_PROCESS_ROUTE_HAS_NO_RENDEZVOUS')


def query(args,until):
    left=until-time.monotonic()
    need(left>0,'BOUNDED_FRESH_PHYSICAL_ADMISSION_DEADLINE')
    done=subprocess.run(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S',*args],
        capture_output=True,timeout=min(left,6))
    need(done.returncode==0 and len(done.stdout)<=2<<20 and len(done.stderr)<=65536,'BOUNDED_HOST_SCHEDULER_READ')
    return done.stdout.decode('utf8').splitlines()


def accounting(identifier,until):
    rows=[line.split('|') for line in query(['sacct','-nP','-j',identifier,'-o',
        'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'],until) if line]
    need(all(len(row)==9 for row in rows),'EXACT_NINE_COLUMN_ACCOUNTING')
    return rows


def verify_approval(a, installed, installation_pin):
    raw=b64decode(a['approval'],validate=True)
    excluded=b64decode(a['exclusion'],validate=True)
    need(0<len(raw)<=8192 and 0<len(excluded)<=8192 and pin(raw)['SHA256']==a['approval_SHA256'],
         'EXACT_ROOT_APPROVAL_BYTES')
    root,ex=json.loads(raw),json.loads(excluded)
    pub=installed['publication']
    now=time.time()
    need(root['request_id']==REQUEST and root['parent']==PARENT and root['node']==NODE
         and root['publication']==pub and root['source_manifest_SHA256']==MF_SHA
         and root['stage_install_SHA256']==installation_pin['SHA256']
         and root['prior_3660_receipt_SHA256']==CPU_SHA and root['prior_3640_archive_SHA256']==GPU_ARCHIVE
         and all(root[k] is True for k in ('root_sole_owner','one_DCU_once','no_prior_unknown_conflict',
             'fresh_prefix7_diagnostic_GO','current_GPU_capability_gate_before_any_updates','no_replay'))
         and root['not_before_epoch']<=now<=root['not_after_epoch']
         and root['not_after_epoch']-root['not_before_epoch']<=180
         and root['exclude_through_epoch']>=root['not_after_epoch']+WHOLE
         and root['ordinary_exclusion_SHA256']==pin(excluded)['SHA256'], 'EXACT_CURRENT_ROOT_PREFIX_AND_PUBLICATION')
    need(ex['scope']==REQUEST and ex['node']==NODE and ex['parent']==PARENT and ex['published_commit']==pub
         and ex['root_only'] is ex['ordinary_exclusion_until_independent_post'] is True
         and ex['no_other_n13_active_or_unknown'] is True and ex['exclude_through_epoch']==root['exclude_through_epoch']
         and re.fullmatch('[0-9a-f]{64}',ex['root_ordinary_custody_proof_SHA256']),
         'REAL_ROOT_ORDINARY_NODE_EXCLUSION_REQUIRED')
    return root,pin(raw),pin(excluded)


def bind(root, approval, exclusion, install, worker, row, started, until, login):
    from contract import COMPUTE_MODEL_STAMP,host_model_identity,validate_preflight,validate_assets
    from placement import step_tres,utc,bind_authority
    now=time.time()
    state=worker['actual_physical']
    guest=worker['container_observation']
    stamp=worker['compute_host_model_identity']
    need(row[:4]==[worker['step'],CPU_LABEL,'COMPLETED','0:0'] and row[5]==NODE and step_tres(row[6],'pre')
         and re.fullmatch(PARENT+r'\.[0-9]+',row[0]) and row[4].isdecimal() and int(row[4])<=60
         and started-1<=utc(row[7])<=utc(row[8])<=now+1 and time.monotonic()<until
         and 0<=now-worker['observed_epoch']<=30 and worker['zero_GPU'] is True and worker['passed'] is True
         and state['step']==row[0] and state['node']==NODE and state['kfd_holder_count']==0
         and len(state['pci'])==len(set(state['pci']))==8 and all(worker['phase_results'].values())
         and set(worker['phase_results'])==set(PHASES)
         and worker['container_stdout']['marker_lines']==1 and not worker['container_stdout']['other_nonblank_lines']
         and guest['mount_readonly'] is True and guest['weight_bytes_hashed']==0
         and stamp==guest['file_identity']==COMPUTE_MODEL_STAMP and host_model_identity(login)
         and login[1:]==stamp[1:], 'EXACT_FRESH_PASSIVE_N13_CPU_ACCOUNTING_AND_RO_MODEL')
    epoch=time.time()
    route=dict(scope=REQUEST,parent=PARENT,node=NODE,boot_id=state['boot_id'],pci=state['pci'],
         observed_epoch=worker['observed_epoch'],exact_actual_physical_step=row[0],device_architecture='gfx936',
         passive_inventory_complete=True,kfd_holder_count=0,same_UID_drm_or_kfd_fd_owner_count=0,
         actual_login_model_identity=login,actual_readonly_mount_model_identity=stamp,
         current_physical_receipt_SHA256=pin(encode(worker))['SHA256'],root_intent_SHA256=approval['SHA256'],
         one_process_no_rendezvous=True,expected_compute_UID_verified=True,
         same_image_provider=True,runtime_single_DCU_path_qualified=True)
    route_raw=encode(route)
    auth=dict(request_id=REQUEST,parent=PARENT,node=NODE,publication=root['publication'],epoch=epoch,
        observed_epoch=epoch,boot_id=state['boot_id'],valid_through_epoch=root['exclude_through_epoch'],
        root_sole_owner=True,one_DCU_once=True,no_prior_unknown_conflict=True,
        ordinary_node_exclusion_until_independent_post=True,qualified_original_unprofiled_route=True,
        fresh_prefix7_diagnostic_GO=True,current_GPU_capability_gate_before_any_updates=True,
        prior_3660_receipt_SHA256=CPU_SHA,prior_3640_archive_SHA256=GPU_ARCHIVE,
        physical_preflight_SHA256=pin(route_raw)['SHA256'],root_intent_SHA256=approval['SHA256'],
        ordinary_exclusion_evidence_SHA256=exclusion['SHA256'])
    pre=json.loads(PREFLIGHT)
    pre['model']['host_file_identity']=login
    pre['model']['file_identity']=stamp
    pre['native_source_manifest']['path']=str(SRC/'NATIVE_BINDING.json')
    pre.update(run=str(RUN),request_id=REQUEST,epoch=epoch,deadline=epoch+WHOLE,preflight_stage_install_SHA256=install['SHA256'])
    validate_preflight(pre,epoch)
    validate_assets(pre,execution_mount=False)
    bind_authority(auth,request=REQUEST,publication=root['publication'],epoch=epoch,
                   route=dict(route,receipt_SHA256=auth['physical_preflight_SHA256']))
    return epoch,{'preflight.json':encode(pre),'placement-authority.json':encode(auth),'physical-route.json':route_raw}


def controller_argv(epoch, publication, pins):
    return [PYTHON,'-B','-u',str(SRC/'controller.py'),'--workspace',str(RUN),'--epoch',str(epoch),
        '--preflight-sha256',pins['preflight.json']['SHA256'],
        '--authority-sha256',pins['placement-authority.json']['SHA256'],
        '--route-sha256',pins['physical-route.json']['SHA256'],'--publication',publication,'--source-manifest-sha256',MF_SHA]


def owner(a):
    need(a['request']==REQUEST and a['physical_request']==CPU_REQUEST,'EXACT_ONE_NEW_PREFIX_AUTHOR_AND_PHYSICAL_SCOPE')
    no_ports(os.environ)
    installed,install=check_install(a['installed_SHA256'])
    root,approval,exclusion=verify_approval(a,installed,install)
    lock=R/'incoming'/(LABEL+'-owner-'+REQUEST)
    need(not any(os.path.lexists(RUN/name) for name in PRIVATE_NAMES|{'intent.json','outcome.json','outputs','miopen-private'}),
         'NEW_OWNER_EXCLUSIVE_PRIVATE_NAMESPACE')
    lock.mkdir(mode=0o700)
    write(lock/'admission-intent.json',encode(dict(request=REQUEST,physical_request=CPU_REQUEST,stage=install,
        authority=approval,ordinary_exclusion=exclusion)))
    sys.path.insert(0,str(SRC))
    from placement import live_views,shared_lock
    from contract import WEIGHT,stamp,host_model_identity
    from profile_gate import cpu_prerequisite,previous_3640
    cpu_prerequisite()
    previous_3640()
    started=time.time()
    until=time.monotonic()+45
    same_window=lambda args:query(args,until)
    with shared_lock(R):
        need(live_views(same_window,os.getuid(),started),'ACTUAL_CURRENT_DUAL_VIEW_EMPTY_BEFORE_CPU')
        need(not any(row[1]==CPU_LABEL for row in accounting(PARENT,until)),'NO_EXISTING_NEW_CPU_LABEL')
    original=[row for row in accounting('1103606.3637',until) if row[0]=='1103606.3637']
    need(original==[ORIGINAL_3637],'EXACT_ARCHIVED_PREDECESSOR_FRAMED_CPU_TERMINAL')
    weight=Path(WEIGHT)
    login=stamp(weight.lstat())
    need(weight.resolve(strict=True)==weight and host_model_identity(login),'CURRENT_LOGIN_WEIGHT_STABLE_METADATA')
    cmd=['srun','--jobid='+PARENT,'--exact','--nodes=1','--ntasks=1','--nodelist='+NODE,
        '--immediate=5','--cpus-per-task=1','--mem=1024M','--gres=none','--time=00:01:00',
        '--job-name='+CPU_LABEL,PYTHON,'-I','-S','-B','-u','-c',GUARDED_WORKER,str(SRC),CPU_LABEL,str(lock)]
    write(lock/'cpu-intent.json',encode(dict(request=CPU_REQUEST,command_SHA256=pin(encode(cmd))['SHA256'],epoch=started)))
    done=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,timeout=25)
    need(len(done.stdout)<=16384 and len(done.stderr)<=65536,'BOUNDED_PHYSICAL_CPU_PROCESS')
    write(lock/'cpu.stdout.private',done.stdout)
    write(lock/'cpu.stderr.private',done.stderr)
    try: worker=json.loads(done.stdout)
    except (ValueError,UnicodeError): worker={}
    step=worker.get('step',PARENT)
    need(step==PARENT or re.fullmatch(PARENT+r'\.[0-9]+',step),'VALID_CPU_STEP_NUMBER')
    exact=[]
    for _ in range(8):
        exact=[row for row in accounting(step,until) if row[1]==CPU_LABEL]
        need(len(exact)<=1,'ONE_CURRENT_CPU_NUMERIC_STEP')
        if exact and exact[0][2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            break
        time.sleep(.25)
    write(lock/'cpu-terminal.json',encode(dict(request=CPU_REQUEST,rows=exact,rc=done.returncode,
        stdout=pin(done.stdout),stderr=pin(done.stderr),worker_passed=worker.get('passed') is True)))
    need(len(exact)==1 and done.returncode==0 and worker.get('passed') is True,'ORIGINAL_CPU_FAILED_NO_CONTROLLER')
    with shared_lock(R):
        need(live_views(same_window,os.getuid(),started),'ACTUAL_CURRENT_DUAL_VIEW_EMPTY_AFTER_CPU')
    verify_approval(a,installed,install)
    need(stamp(weight.lstat())==login,'UNCHANGED_LOGIN_WEIGHT_DURING_ADMISSION')
    epoch,objects=bind(root,approval,exclusion,install,worker,exact[0],started,until,login)
    pins={name:write(RUN/name,raw) for name,raw in objects.items()}
    command=controller_argv(epoch,installed['publication'],pins)
    write(lock/'controller-intent.json',encode(dict(request=REQUEST,epoch=epoch,pins=pins,
         argv_SHA256=pin(encode(command))['SHA256'],ordinary_exclusion=exclusion)))
    syncdir(lock)
    syncdir(RUN)
    with (lock/'controller.stdout.private').open('xb') as out,(lock/'controller.stderr.private').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
    result=dict(request=REQUEST,scope=REQUEST,PID=child.pid,startticks=int(fields[19]),epoch=epoch,
        deadline=epoch+WHOLE,source_manifest=MF_SHA,preflight_SHA256=pins['preflight.json']['SHA256'],
        physical_step=exact[0][0],physical_request=CPU_REQUEST,replay_allowed=False,
        state='EXACT_PREFIX7_PRODUCTION_CONTROLLER_CHILD_STARTED_OPTIMIZER_CAPABILITY_PENDING')
    write(lock/'owner.json',encode(result))
    syncdir(lock)
    return result


def main():
    need(os.getuid()==EXPECTED_UID and R.resolve(strict=True)==R,'EXACT_AUTHENTICATED_PROJECT_UID')
    os.umask(0o077)
    a=json.load(sys.stdin)
    mode=a['mode']
    call={'stage':stage,'once':owner,'terminal':terminal}[mode]
    print(json.dumps(call(a),sort_keys=True),flush=True)


if __name__=='__main__':
    main()
