"""Build one standalone device translation unit with two fixed head-scale arms."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time


D=Path(__file__).resolve().parent
PLAN=json.loads((D/'PLAN.json').read_bytes())
EXPECTED={'deep_c4_head_scale_i1_v1','deep_c4_head_scale_i4_v1'}


def need(ok, label):
    if not ok:
        raise ValueError(label)


def pin(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_paths(raw):
    target,colon,spec=raw.replace('\\\n',' ').partition(':')
    paths=set(shlex.split(spec))
    need(bool(target and colon and paths) and all(path.startswith('/') for path in paths),'FRESH_M_FORMAT')
    return paths


def main():
    epoch=float(os.environ['TUNING_CPU_EPOCH'])
    deadline=float(os.environ['TUNING_CPU_DEADLINE'])
    job=os.environ['SLURM_JOB_ID']
    step=os.environ['SLURM_STEP_ID']
    need(job==PLAN['parent'] and step.isdecimal() and socket.gethostname().split('.')[0]==PLAN['node'],
         'ORIGINAL_SLURM_PLACEMENT')
    need(0 <= time.time()-epoch < 270 and deadline==epoch+300, 'ORIGINAL_HOST_BUDGET')
    need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
         and os.environ.get('ROCM_PATH')=='/opt/dtk'
         and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'CPU_ONLY_ENVIRONMENT')
    manifest=D/'SOURCE_FILES.json'
    need(pin(manifest)==os.environ['TUNING_CPU_MANIFEST_SHA'],'SOURCE_MANIFEST')
    manifest_pins=json.loads(manifest.read_bytes())['files_SHA256']
    need(all(Path(name).name==name and pin(D/name)==value for name,value in manifest_pins.items()), 'SOURCE_BYTES')
    output=Path('/refactor/candidates')/PLAN['label']
    output.mkdir(mode=0o700)
    started=time.monotonic()
    report=dict(original_step=job+'.'+step,node=PLAN['node'],epoch=epoch,deadline=deadline,GRES='none',
                planned_GPU_translation_units=1,actual_GPU_translation_units=0,actual_GPU_steps=0,
                planned_variants=PLAN['variant_ids'],explicit_Runtime_calls=0,commands=[],
                passed=False,completed=False)
    def save():
        work=output/'CPU-result.tmp'
        work.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        os.replace(work,output/'CPU-result.json')

    def command(name,argv,max_seconds):
        left=min(deadline-time.time()-12,250-(time.monotonic()-started))
        need(left>1,'BOUNDED_BUILD')
        entry=dict(label=name,argv=argv)
        report['commands'].append(entry)
        save()
        process=subprocess.run(argv,capture_output=True,text=True,timeout=min(left,max_seconds))
        for stream in ('stdout','stderr'):
            f=output/(name+'.'+stream)
            f.write_text(getattr(process,stream))
            entry[stream+'_SHA256']=pin(f)
        entry['rc']=process.returncode
        save()
        need(process.returncode==0,name+'_FAILED')
        return process.stdout

    source=str(D/'head_scale.cu')
    obj=str(output/'head_scale.o')
    before=output/'head_scale.before.d'
    md=output/'head_scale.d'
    try:
        for tool,entry in PLAN['tools'].items():
            need(str(Path(tool).resolve(strict=True))==entry['canonical'] and pin(tool)==entry['sha256'],
                 'DTK_TOOLCHAIN_PIN')
        provenance=Path(PLAN['producer_receipt'])
        need(provenance.stat().st_size==PLAN['producer_bytes'] and pin(provenance)==PLAN['producer_SHA256'],
             'DEPENDENCY_PRODUCER_RECEIPT')
        prior=json.loads(provenance.read_bytes())
        need(prior['actual_CPU']=='910407.1113' and prior['completed'] and prior['passed']
             and prior['all19_fresh_M_before_compile'] and prior['all19_MD_equal_fresh_M'],
             'DEPENDENCY_PRODUCER_QUALIFICATION')
        available={}
        for dependencies in prior['fresh_dependencies_SHA256'].values():
            for path,h in dependencies.items():
                need(path not in available or available[path]==h,'INCONSISTENT_PRODUCER_DEPENDENCY_PIN')
                available[path]=h
        need(available.get(PLAN['cuda_runtime_header'])==PLAN['cuda_runtime_header_SHA256'], 'CUDA_HEADER_PIN')
        need(pin(source)==PLAN['source_SHA256']==manifest_pins['head_scale.cu'],'SOURCE_IDENTITY')
        available[source]=PLAN['source_SHA256']
        flags=PLAN['compiler_prefix']
        command('head-scale-M',flags+['-M','-MF',str(before),'-MT',obj,source],70)
        paths=dependency_paths(before.read_text())
        need(source in paths and PLAN['cuda_runtime_header'] in paths
             and all(available.get(p)==pin(p) for p in paths), 'FRESH_M_DEPENDENCY_PIN_DRIFT')
        report.update(all_fresh_M_before_compile=True,before_M_SHA256=pin(before),
                      dependencies_SHA256={p:available[p] for p in sorted(paths)})
        command('head-scale-compile',flags+['-MD','-MF',str(md),'-c',source,'-o',obj],100)
        need(dependency_paths(md.read_text())==paths and all(pin(p)==available[p] for p in paths),'COMPILED_MD_DRIFT')
        text=command('head-scale-symbols',['/usr/bin/nm','-g','--defined-only',obj],20)
        symbols={r[-1] for s in text.splitlines() if len(r:=s.split())>=3 and r[-2]=='T'
                 and r[-1].startswith('deep_c4_head_scale_')}
        need(symbols==EXPECTED,'TWO_CABI_ARM_SYMBOLS')
        maps=Path('/proc/self/maps').read_text()
        need(not re.search(r'lib(?:infiniops|infinirt|infiniccl|infinilm_runtime|torch|amdhip64|galaxyhip)',maps),
             'NO_EXPLICIT_RUNTIME_IN_CONTROLLER')
        report.update(passed=True,actual_GPU_translation_units=1,object=obj,object_SHA256=pin(obj),
                      MD_SHA256=pin(md),exports=sorted(symbols),device_qualified=False)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report.update(error_kind=type(error).__name__,failed_check=str(error)[:240],
                      gpu_TU_compile_command_attempted=any(c['label']=='head-scale-compile' for c in report['commands']))
    finally:
        report.update(completed=True,elapsed_seconds=time.monotonic()-started)
        save()
    print(json.dumps({k:report.get(k) for k in ('original_step','passed','object_SHA256','failed_check')}),
          flush=True)
    return int(not report['passed'])


if __name__=='__main__':
    raise SystemExit(main())
