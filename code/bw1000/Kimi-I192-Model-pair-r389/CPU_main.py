"""One bounded Linux CPU entry; no Native load, weights, GPU or real model job."""
from pathlib import Path
import ast,datetime,hashlib,json,os,sys,time,traceback

D=Path(__file__).resolve().parent
OUT=Path('/refactor/candidates/Kimi-I192-Model-pair-r389')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights=False,
           actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
           source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        assert q['source_manifest_SHA256']==os.environ['KIMI_R389_CPU_SOURCE_SHA']
        for key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:assert os.environ.get(key)==''
        assert not os.environ.get('LD_PRELOAD')
        manifest=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
        for name,digest in manifest['files_SHA256'].items():assert sha(D/name)==digest,name
        for path in D.rglob('*.py'):ast.parse(path.read_text())
        for path in D.rglob('*.sh'):assert b'\r\n' not in path.read_bytes()
        from source_proof import verify
        q['dynamic_dispatch_source']=verify()
        import check_CPU
        q['checks']=check_CPU.main()
        from audit import verify_artifacts
        from check_entry_CPU import FaultGate,environment
        from lifecycle import Settings
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'control').mkdir()
            settings=Settings(environment(root),root)
            bound=verify_artifacts(D/'target',FaultGate(settings))
            q['actual327_8df_source_and_receipt_binding']=bound['audit']
            frozen=json.loads((bound['driver']/'frozen_inputs.json').read_bytes())
            from spec_driver import _validate_fixtures
            retained=_validate_fixtures(frozen)['retained8635']
            q['actual_frozen_input']={k:retained[k] for k in ['name','input_tokens','prompt_ID_SHA256','output_budget','eos']}
        assert 'torch'not in sys.modules and 'infinilm'not in sys.modules
        q.update(completed=True,passed=True)
    except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start)
        (OUT/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
        for child,field in [('target','source_manifest_SHA256'),('controller','controller_manifest_SHA256')]:
            out=OUT/child;out.mkdir();receipt=dict(q)
            receipt[field]=sha(D/child/'SOURCE_MANIFEST.json')
            (out/'CPU-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:q.get(k) for k in ['completed','passed','actual_CPU','elapsed_s','error']}),flush=True)
    return 0 if q['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
