"""Attribute a MIOpen library in the actual T09 link environment without importing it."""
import argparse
import ast
from hashlib import sha256
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

REQUEST='a5469fc7828a4de98f61de867abda4cd'
PARENT,NODE='1103606','f13r3n13'
HISTORIC_SHA='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba'
HISTORIC_SIZE=18705784
READ_ELF='/usr/bin/readelf'
DANGEROUS_LD=('LD_PRELOAD','LD_AUDIT','LD_PROFILE')
VARS=('MIOPEN_USER_DB_PATH','MIOPEN_SYSTEM_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR',
      'MIOPEN_CACHE_DIR','MIOPEN_LOCKFILE_DIR','MIOPEN_LOCKFILE_PATH','MIOPEN_FIND_MODE',
      'MIOPEN_DEBUG_DISABLE_FIND_DB','MIOPEN_DISABLE_CACHE','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME')
TOKENS=VARS+('miopen-lockfiles','.config/miopen','.cache/miopen','miopenGetVersion','src/lock_file.cpp',
              'MIOPEN_USER_DB_PATH_ENV','MIOPEN_SYSTEM_DB_PATH_ENV')
NUM=re.compile(r'libMIOpen\.so(?:\.[0-9]+)*')
PROVIDER=('/opt/dtk/','/opt/rocm/','/usr/local/lib/','/training-runtime/','/.singularity.d/libs/')
MAX_ELF=256<<20
MAX_TEXT=12<<20


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return sha256(raw).hexdigest()


def allowed_path(path):
    value=str(path)
    return value if value.startswith(PROVIDER) else None


def identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def run(argv,deadline):
    remaining=deadline-time.monotonic()
    require(remaining>0,'ORIGINAL_EPHEMERAL_CPU_BUDGET')
    env=dict(os.environ,LC_ALL='C')
    result=subprocess.run(argv,timeout=min(8,remaining),stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    require(len(result.stdout)<=MAX_TEXT and len(result.stderr)<=65536,'BOUNDED_TOOL_OUTPUT')
    return result.returncode,result.stdout


def elf(trace,path,kind):
    rc,raw=trace([READ_ELF,kind,'--',str(path)])
    require(rc==0,'ELF_METADATA_UNAVAILABLE')
    return raw


def matches(raw):
    text=raw.decode('utf8','strict')
    rows=[]
    unresolved=[]
    for line in text.splitlines():
        name,sep,right=line.strip().partition(' => ')
        if not sep or not NUM.fullmatch(name):
            continue
        candidate=right.split(' (',1)[0]
        if candidate=='not found':
            unresolved.append(name)
        else:
            require(candidate.startswith('/'),'LIBRARY_TRACE_ABSOLUTE_PATH')
            rows.append((name,candidate))
    return rows,unresolved


def scan_elf(path,deadline):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=MAX_ELF,'EXACT_BOUNDED_LIBRARY_FILE')
    count=0
    digest=sha256()
    tail=b''
    found=set()
    with path.open('rb') as stream:
        require(identity(os.fstat(stream.fileno()))==identity(before),'LIBRARY_CHANGED_AT_OPEN')
        while True:
            require(time.monotonic()<deadline,'ORIGINAL_EPHEMERAL_CPU_BUDGET')
            chunk=stream.read(1<<20)
            if not chunk:break
            count+=len(chunk)
            digest.update(chunk)
            raw=tail+chunk
            found.update(word for word in TOKENS if word.encode() in raw)
            tail=raw[-256:]
        require(identity(os.fstat(stream.fileno()))==identity(before),'LIBRARY_CHANGED_IN_SHA_READ')
    require(identity(path.lstat())==identity(before) and count==before.st_size,'LIBRARY_CHANGED_AFTER_SHA_READ')
    return dict(bytes=count,SHA256=digest.hexdigest(),compiled_markers={word:word in found for word in TOKENS})


def build_id(raw):
    ids=set(re.findall(rb'Build ID:\s*([0-9a-f]{8,128})(?:\s|$)',raw))
    require(len(ids)==1,'ONE_ACTUAL_GNU_BUILD_ID')
    return next(iter(ids)).decode()


def lock_family_symbols(raw):
    result=set()
    for line in raw.decode('utf8','replace').splitlines():
        atom=line.split()
        if not atom or not re.fullmatch(r'[_A-Za-z0-9@.]{2,260}',atom[-1]):
            continue
        name=atom[-1]
        if re.search(r'LockFile|GetUserDb|GetSystemDb|GetCachePath|GetCacheDir|GetHomeDir|miopenGetVersion',name):
            result.add(name)
    require(len(result)<=96,'BOUNDED_LOCK_SYMBOLS')
    return sorted(result)


def header_markers():
    values={}
    candidates=(Path('/opt/dtk/include/miopen/version.h'),Path('/opt/dtk/include/miopen/miopen_version.h'),
                Path('/opt/dtk/lib/cmake/miopen/miopen-config-version.cmake'))
    for path in candidates:
        try: info=path.stat()
        except FileNotFoundError: continue
        require(stat.S_ISREG(info.st_mode) and info.st_size<=65536,'BOUNDED_INSTALLED_VERSION_METADATA')
        raw=path.read_bytes()
        require(len(raw)==info.st_size,'VERSION_METADATA_CHANGED')
        macros=re.findall(rb'(?m)^\s*#\s*define\s+(MIOPEN_VERSION_(?:MAJOR|MINOR|PATCH))\s+([0-9]+)',raw)
        fields={name.decode(): version.decode() for name,version in macros}
        package=re.findall(rb'(?m)^\s*set\(PACKAGE_VERSION\s+"([0-9.]+)"',raw)
        if len(package)==1:
            fields['PACKAGE_VERSION']=package[0].decode()
        values[str(path)]={'SHA256':sha(raw),'bytes':len(raw),'version_literals':fields}
    return values


def directory_shape(value, protected_home):
    if not value:
        return dict(configured=False)
    path=Path(value)
    kind=('PROTECTED_HOME' if path.is_absolute() and path.is_relative_to(protected_home)
          else 'INSTALLED_RUNTIME' if allowed_path(path) else 'OTHER_OR_RELATIVE')
    result=dict(configured=True,category=kind,path_SHA256=sha(str(path).encode()))
    if kind=='OTHER_OR_RELATIVE':
        return result
    try:
        info=path.lstat()
        result.update(present=True,is_directory=stat.S_ISDIR(info.st_mode),symlink=stat.S_ISLNK(info.st_mode))
        if not result['symlink']:
            result['statvfs_readonly']=bool(os.statvfs(path).f_flag&os.ST_RDONLY)
    except FileNotFoundError:
        result['present']=False
        try: result['parent_statvfs_readonly']=bool(os.statvfs(path.parent).f_flag&os.ST_RDONLY)
        except OSError as error: result['parent_errno']=error.errno
    except OSError as error:
        result['errno']=error.errno
    return result


def locate(report,deadline):
    trace=lambda argv:run(argv,deadline)
    require(not any(os.environ.get(key) for key in DANGEROUS_LD),'UNEXPECTED_LOADER_SIDE_EFFECT_ENV')
    sys_preload=Path('/etc/ld.so.preload')
    if sys_preload.is_file():
        with sys_preload.open('rb') as stream:
            raw=stream.read(4097)
        require(len(raw)<=4096 and not any(line.split(b'#',1)[0].strip() for line in raw.splitlines()),
                'NONEMPTY_SYSTEM_LOADER_PRELOAD')
    spec=PathFinder.find_spec('torch',sys.path)
    require(spec is not None and spec.origin is not None,'ACTUAL_TORCH_PACKAGE_PROVENANCE')
    torch_root=Path(spec.origin).resolve(strict=True).parent
    require(allowed_path(torch_root) is not None and 'torch' not in sys.modules,'EXPECTED_TORCH_PATH_WITHOUT_IMPORT')
    with (torch_root/'version.py').open('rb') as stream:
        torch_version=stream.read(65537)
    require(len(torch_version)<=65536,'BOUNDED_TORCH_VERSION_SOURCE')
    literals={}
    for node in ast.parse(torch_version).body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,(str,type(None))):
                for target in targets:
                    if isinstance(target,ast.Name) and target.id in ('__version__','hip','cuda','git_version'):
                        if node.value.value is None or re.fullmatch(r'[A-Za-z0-9_.+:-]{1,128}',node.value.value):
                            literals[target.id]=node.value.value
    report['torch_package']=dict(path=str(torch_root),version_source_SHA256=sha(torch_version),literals=literals,imported=False)
    report['phase']='LOADER_RESOLUTION'
    exec_elf=Path('/proc/self/exe').resolve(strict=True)
    interp=re.findall(rb'Requesting program interpreter:\s*([^\]\r\n]+)',elf(trace,exec_elf,'-lW'))
    require(len(interp)==1,'EXACT_PYTHON_ELF_INTERPRETER')
    loader=Path(interp[0].decode()).resolve(strict=True)
    require(str(loader).startswith(('/lib/','/lib64/','/usr/lib/')) and 'ld-linux' in loader.name,
            'EXPECTED_TRUSTED_GLIBC_LINKER')
    core=[*sorted(torch_root.glob('_C*.so')),torch_root/'lib/libtorch_hip.so']
    require(len(core)<=5 and any(path.is_file() and path.name.startswith('_C') for path in core),
            'BOUNDED_TORCH_EXTENSION_FILES')
    selected={}
    projections={}
    for path in core:
        if not path.is_file():continue
        target=path.resolve(strict=True)
        require(target.is_relative_to(torch_root),'UNEXPECTED_TORCH_DSO_ESCAPE')
        dynamic=elf(trace,target,'-dW')
        all_needed=[match.decode() for match in re.findall(rb'NEEDED.*?\[([-.A-Za-z0-9_+]{1,150})\]',dynamic)]
        rc,raw=trace([str(loader),'--list',str(target)])
        rows,missing=matches(raw)
        canonical=[]
        for name,resolved in rows:
            resolved=Path(resolved)
            actual=resolved.resolve(strict=True)
            require(allowed_path(actual) is not None and actual.is_file(),'MIOPEN_RESOLUTION_OUTSIDE_APPROVED_LIB_ROOT')
            selected[str(actual)]=actual
            canonical.append(dict(soname=name,path=allowed_path(resolved),real_path=str(actual)))
        projections[path.name]=dict(loader_rc=rc,DSO_miopen_edges=[item for item in all_needed if NUM.fullmatch(item)],
                                    miopen_resolutions=canonical,unresolved_miopen=missing,
                                    resolver_stdout_SHA256=sha(raw),loader_trace_no_constructors=True)
    report['resolutions']=projections
    require(len(selected)==1 and all(not item['unresolved_miopen'] for item in projections.values())
            and any(item['loader_rc']==0 and item['miopen_resolutions'] for item in projections.values()),
            'MIOPEN_LOADER_ROUTE_UNRESOLVED_OR_AMBIGUOUS')
    target=next(iter(selected.values()))
    report['phase']='ELF_BINARY_IDENTITY'
    binary=scan_elf(target,deadline)
    dynamic=elf(trace,target,'-dW')
    names=re.findall(rb'SONAME.*?\[(libMIOpen\.so(?:\.[0-9]+)*)\]',dynamic)
    require(len(names)==1,'ONE_EXACT_MIOPEN_ELF_SONAME')
    binary.update(real_path=str(target), GNU_build_id=build_id(elf(trace,target,'-nW')),
                  ELF_SONAME=names[0].decode(),
                  relevant_symbols=lock_family_symbols(elf(trace,target,'-Ws')))
    binary['matches_historic_other_workload_candidate']=binary['SHA256']==HISTORIC_SHA and binary['bytes']==HISTORIC_SIZE
    report['MIOpen']=binary
    report['installed_version_candidates']=header_markers()
    report['phase']='PATH_OBSERVATION'
    protected=Path('/public/home/aclsylqidf')
    env=dict()
    for key in VARS:
        if key.endswith(('_PATH','_DIR','_HOME')):
            env[key]=directory_shape(os.environ.get(key),protected)
        else:
            env[key]=dict(configured=bool(os.environ.get(key)))
    report['variables_observed']=env
    report['HOME_matches_original_protected_value']=os.environ.get('HOME')==str(protected)
    report['original_default_observations']={name:directory_shape(str(protected/path),protected) for name,path in {
        'lock_directory':'.config/miopen/miopen-lockfiles','default_config_candidate':'.config/miopen',
        'default_cache_candidate':'.cache/miopen'}.items()}
    report.update(phase='RESOLVED',state='EXACT_MIOPEN_LINK_AND_ELF_IDENTITY_OBSERVED',
                  binary_markers_prove_lock_DB_cache_independence=False,lock_redirection_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    report=dict(request=REQUEST,phase='ENTRY',state='BLOCKED',actual_GPU_SDK_calls=0,actual_model_loads=0,
                actual_db_cache_lock_writes=0,planned_optimizer_updates=0,
                binary_markers_prove_lock_DB_cache_independence=False)
    try:
        step=os.environ.get('SLURM_STEP_ID','')
        require(os.environ.get('SLURM_JOB_ID')==PARENT and step.isdecimal()
                and all(os.environ.get(key)=='' for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
                and sys.dont_write_bytecode and 0<args.deadline-time.time()<=120,'EXACT_BOUNDED_ZERO_GRES_ENTRY')
        report['step']=PARENT+'.'+step
        locate(report,time.monotonic()+args.deadline-time.time())
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as error:
        value=str(error)
        report.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                      error_code=value if type(error) is ValueError and re.fullmatch('[A-Z0-9_]{1,100}',value) else None)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report['phase']=='RESOLVED' else 2


if __name__=='__main__':
    raise SystemExit(main())
