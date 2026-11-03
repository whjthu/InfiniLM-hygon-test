"""One bounded ELF/source handoff job. ldd trace only: never execute a DSO."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from check_CPU import check_stage

D=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Kimi-private-loading-Runtime-integration-r442-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    B.mkdir(exist_ok=False)
    q=dict(completed=False,passed=False,Native_dlopen=False,Runtime_executed=False,
           GPU=False,Model_READY=False,compile=0,link=0,commands=[])
    start=time.monotonic()
    def run(argv,name,env=None):
        left=250-(time.monotonic()-start);assert left>0
        t=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=min(40,left),env=env)
        (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
        q['commands'].append(dict(argv=argv,name=name,rc=p.returncode,wall_s=time.monotonic()-t,
            stdout_SHA256=sha(B/(name+'.stdout')),stderr_SHA256=sha(B/(name+'.stderr'))))
        assert p.returncode==0,(name,p.stderr[-2000:])
        return p.stdout
    try:
        q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        assert all(os.environ[k]=='-1' for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        assert sha(D/'SOURCE_MANIFEST.json')==os.environ['R442_SOURCE_SHA256']
        for name,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/name)==h,name
        c=json.loads((D/'CPU_PLAN.json').read_bytes())
        for p,h in c['pinned_files_SHA256'].items():assert sha(p)==h,p
        q['consumer']=check_stage(Path(c['parent']),Path(c['text']),Path(c['config']),Path(c['candidate']),B/'deployment')
        deployed=q['consumer']['proof'];private=Path(deployed['runtime_directory'])/'libinfinilm_runtime.so'
        env=dict(os.environ, LD_LIBRARY_PATH=str(private.parent)+':'+os.environ['LD_LIBRARY_PATH'])
        # ldd traces do not inherit the Python executable's exported Py* symbols.
        # Supply precisely the already qualified libpython to that TRACE only.
        assert sha(c['libpython'])==c['libpython_SHA256']
        env['LD_PRELOAD']=c['libpython']
        for name,p in c['ELF'].items():
            dynamic=run(['readelf','-d',p],name+'-dynamic')
            needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
            if name in ['Native','Common_extension']:assert 'libinfinilm_runtime.so' in needed and all('/'not in s for s in needed if 'infinilm_runtime' in s)
            if name=='Common_extension':assert '(RPATH)'not in dynamic and '$ORIGIN/infinilm/_lib' in dynamic
            resolved=run(['ldd','-r',p],name+'-ldd',env)
            assert 'not found'not in resolved and 'undefined symbol'not in resolved,(name,resolved)
            if name in ['Native','Common_extension']:
                assert re.search(r'libinfinilm_runtime\.so\s+=>\s+'+re.escape(str(private))+r'\s',resolved),resolved
                runtime_lines=[line for line in resolved.splitlines()if 'libinfinilm_runtime' in line]
                assert len(runtime_lines)==1 and str(private)in runtime_lines[0],('multiple Runtime dependencies',runtime_lines)
                loaded={}
                for line in resolved.splitlines():
                    match=re.search(r'^\s*(\S+)\s+=>\s+(/\S+)\s',line)
                    if match:loaded[match.group(1)]=match.group(2)
                m=json.loads(Path(json.loads(Path(deployed['config']).read_bytes())['model_manifest']).read_bytes())
                for basename,digest in m['loaded_library_SHA256'].items():
                    assert basename in loaded and sha(loaded[basename])==digest,(name,basename,loaded.get(basename))
                q.setdefault('actual_trace_providers',{})[name]={base:dict(path=loaded[base],SHA256=sha(loaded[base]))for base in m['loaded_library_SHA256']}
        old=run(['nm','-D','--defined-only',c['baseline']],'old-exports')
        new=run(['nm','-D','--defined-only',str(private)],'new-exports')
        strong=lambda t:{line.split()[-1] for line in t.splitlines() if len(line.split())==3 and line.split()[1]in ['T','D','B','R','S','G']}
        assert strong(old)<=strong(new)
        assert 'modern_cpu_copy_row_hits_r436' in strong(new)
        q.update(passed=True,Model_GO=False,actual_ELF_trace_uses_one_candidate=True,
            strong_exports_preserved=len(strong(old)),coldfile_IO_and_multiowner_H2D='UNKNOWN')
    except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,wall_s=time.monotonic()-start)
        (B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
        print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','wall_s']}),flush=True)
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
