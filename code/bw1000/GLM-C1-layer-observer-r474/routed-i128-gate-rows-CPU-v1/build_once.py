"""Two fresh-dependency-matched gfx936 translations without loading a runtime."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from variants import ROWS, variant


PLAN = json.loads((D/'PLAN.json').read_bytes())


def need(condition, code):
    if not condition:
        raise ValueError(code)


def pin(path):
    path = Path(path)
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def dependency_paths(text):
    lhs, colon, rhs = text.replace('\\\n',' ').partition(':')
    paths = set(shlex.split(rhs))
    need(bool(lhs and colon and paths) and all(p.startswith('/') for p in paths), 'DEPENDENCY_PARSE')
    return paths


def leaf_exports(symbols):
    return {fields[-1] for row in symbols.splitlines() if len(fields := row.split()) >= 3
            and fields[-2] == 'T' and fields[-1].startswith('glm_r470_i128_')}


def main():
    job = os.environ['SLURM_JOB_ID']
    step = os.environ['SLURM_STEP_ID']
    epoch = float(os.environ['TUNING_CPU_EPOCH'])
    deadline = float(os.environ['TUNING_CPU_DEADLINE'])
    need(job == PLAN['parent'] and step.isdecimal() and socket.gethostname().split('.')[0] == PLAN['node'],
         'SLURM_IDENTITY')
    need(deadline == epoch+300 and 0 <= time.time()-epoch < 270, 'ORIGINAL_BUDGET')
    need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
         and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT')
         and os.environ.get('ROCM_PATH')=='/opt/dtk', 'CPU_ONLY_ENVIRONMENT')
    manifest = D/'SOURCE_FILES.json'
    need(pin(manifest) == os.environ['TUNING_CPU_MANIFEST_SHA'], 'MANIFEST_IDENTITY')
    for name, digest in json.loads(manifest.read_bytes())['files_SHA256'].items():
        need(Path(name).name == name and pin(D/name) == digest, 'SOURCE_IDENTITY')
    output = Path('/refactor/candidates') / PLAN['label']
    output.mkdir(mode=0o700)
    started = time.monotonic()
    report = dict(original_step=job+'.'+step,node=PLAN['node'],epoch=epoch,deadline=deadline,
                  planned_new_device_TUs=len(ROWS),compiled={},commands=[],completed=False,
                  passed=False,actual_GPU_steps=0,SDK_calls=0,weights_read=0,numerical_qualified=False)

    def save():
        temp = output/'CPU-result.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        os.replace(temp,output/'CPU-result.json')

    def execute(name, args, cap):
        left=min(deadline-time.time()-10,257-(time.monotonic()-started))
        need(left>1,'ORIGINAL_BUILD_BUDGET')
        entry=dict(label=name,argv=args)
        report['commands'].append(entry)
        save()
        process=subprocess.run(args,capture_output=True,text=True,timeout=min(left,cap))
        for stream in ('stdout','stderr'):
            file=output/(name+'.'+stream)
            file.write_text(getattr(process,stream))
            entry[stream+'_SHA256']=pin(file)
        entry['rc']=process.returncode
        save()
        need(process.returncode==0,name+'_FAILED')
        return process.stdout

    try:
        for name, value in PLAN['tools'].items():
            need(str(Path(name).resolve(strict=True))==value['canonical'] and pin(name)==value['sha256'],
                 'DTK_TOOLCHAIN_DRIFT')
        p=Path(PLAN['original_receipt'])
        need(p.stat().st_size==PLAN['original_receipt_bytes'] and pin(p)==PLAN['original_receipt_SHA256'],
             'ORIGINAL_CPU1087_RECEIPT')
        original=json.loads(p.read_bytes())
        leaf=original['compiled']['routed_i128.cu']
        need(original['actual_CPU']=='910407.1087' and original['completed'] and original['passed']
             and original['all_fresh_M_equal_MD'] and leaf['object']==PLAN['original_object']
             and leaf['object_SHA256']==pin(leaf['object'])==PLAN['original_object_SHA256'],
             'ORIGINAL_ROUTED_LEAF')
        old_source=PLAN['original_source']
        need(leaf['source_SHA256']==pin(old_source)==PLAN['original_source_SHA256'], 'ORIGINAL_ROUTED_SOURCE')
        raw=Path(old_source).read_bytes()
        original_deps=original['fresh_dependencies_SHA256']['routed_i128.cu']
        need(original_deps.get(old_source)==PLAN['original_source_SHA256'], 'ORIGINAL_SOURCE_DEPENDENCY')
        del original_deps[old_source]
        argv=PLAN['compiler_prefix']
        old_commands=[row['argv'] for row in original['commands'] if row['name']=='routed_i128-compile']
        need(len(old_commands)==1 and old_commands[0][:len(argv)+2]==argv+['-MD','-MF']
             and old_commands[0][-4:]==['-c',old_source,'-o',leaf['object']], 'ORIGINAL_COMPILER_RECIPE')
        expected_exports={'glm_r470_i128_validate','glm_r470_i128_gate','glm_r470_i128_down'}
        nm=['/usr/bin/nm','-g','--defined-only']
        need(leaf_exports(execute('original-symbols',nm+[leaf['object']],15))==expected_exports,
             'ORIGINAL_LEAF_CABI')
        todo=[]
        for rows in ROWS:
            out=output/f'rows{rows}'
            out.mkdir()
            source=str(out/'routed_i128.cu')
            frozen=D/f'rows{rows}.cu'
            need(frozen.read_bytes()==variant(raw,rows), 'ONE_VARIABLE_SOURCE_DRIFT')
            shutil.copyfile(frozen,source)
            candidate_sha=pin(source)
            need(candidate_sha==PLAN['variants'][str(rows)]['source_SHA256'], 'CANDIDATE_SHA_DRIFT')
            expected_dep={**original_deps,source:candidate_sha}
            observed=execute(f'rows{rows}-M',argv+['-M',source],48)
            need(dependency_paths(observed)==set(expected_dep)
                 and all(pin(name)==digest for name,digest in expected_dep.items()), 'FRESH_M_DEPENDENCY_DRIFT')
            todo.append((rows,out,source,expected_dep,candidate_sha))
        report['all_fresh_M_before_compile']=True
        for rows,out,source,deps,candidate_sha in todo:
            md=out/'routed_i128.d'
            obj=str(out/'routed_i128.o')
            execute(f'rows{rows}-compile',argv+['-MD','-MF',str(md),'-c',source,'-o',obj],88)
            need(dependency_paths(md.read_text())==set(deps)
                 and all(pin(name)==digest for name,digest in deps.items()), 'COMPILED_MD_DEPENDENCY_DRIFT')
            need(leaf_exports(execute(f'rows{rows}-symbols',nm+[obj],15))==expected_exports,
                 'CANDIDATE_LEAF_CABI')
            report['compiled'][str(rows)]=dict(candidate_id=PLAN['variants'][str(rows)]['id'],
                source_SHA256=candidate_sha,object=obj,object_SHA256=pin(obj),MD_SHA256=pin(md),
                dependency_count=len(deps),host_wrapper_ABI=sorted(expected_exports))
            save()
        need(pin(leaf['object'])==PLAN['original_object_SHA256'], 'ORIGINAL_BINARY_DRIFT')
        maps=Path('/proc/self/maps').read_text()
        need(not re.search(r'lib(?:infiniops|infinirt|infiniccl|infinilm_runtime|torch|amdhip64|galaxyhip)',maps)
             and 'torch' not in sys.modules,'EXPLICIT_COMPILER_PARENT_RUNTIME_NONE')
        report['passed']=True
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        report.update(error_kind=type(error).__name__,failed_check=str(error)[:240])
    finally:
        report.update(completed=True,elapsed_seconds=time.monotonic()-started)
        save()
    print(json.dumps(dict(step=report['original_step'],passed=report['passed'],
                          variants=sorted(report['compiled']),error=report.get('failed_check'))),flush=True)
    return int(not report['passed'])


if __name__=='__main__':
    raise SystemExit(main())
