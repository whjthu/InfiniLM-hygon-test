"""One pure-HOST changed consumer closure; no encoding/Native/SDK/GPU execution."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
D=Path(__file__).resolve().parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
OUT=R/'candidates/Kimi-wide-fewshot-quality-r448-client-CPU-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not OUT.exists();OUT.mkdir();path=OUT/'CPU-result.json';start=time.monotonic()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,SDK=False,weights=False,Model_READY=False,encoding937_repeated=False,phases=[])
    def save():
        t=path.with_suffix('.tmp');t.write_bytes((json.dumps(q,indent=2,allow_nan=False)+'\n').encode());t.replace(path)
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
        def deny_weights(event,args):
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Changed CPU consumer forbids weight payloads')
        sys.addaudithook(deny_weights)
        import controller
        mf=sha(D/'SOURCE_MANIFEST.json');assert mf==os.environ['KIMI448_CLIENT_SOURCE_SHA256'];m=controller.verify(mf)
        q.update(source_manifest_SHA256=mf,actual_encoding_CPU_receipt_SHA256=m['encoding_CPU_receipt_SHA256'],verified_source_and_model_ELF_bytes_only=True);q['phases'].append('actual_controller_verify_scope_source937_and_parent_metadata_bytes')
        for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
        from preflight import check
        q['final_Bash_and_typedCLI']=check(D,controller.PACKAGE,m);assert q['final_Bash_and_typedCLI']['passed'];q['phases'].append('actual_final_Bash_exports_typedCLI4roles_noServe')
        from check_entry_CPU import check
        q['actual_changed_entry']=check(D,m);assert q['actual_changed_entry']['passed'];q['phases'].append('actual_shared_ready_handoff_publicfactory_clientMain256_cancel_identity_deadline_and_terminalcollector')
        q['passed']=True
    except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start,project_maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if any(v in s for v in ['_infinilm.cpython','libKimi_','libDeep_','libGLM_','libinfinilm_runtime','libinfiniops','libinfinirt','libinfiniccl'])],Torch_imported='torch'in sys.modules)
        if q['project_maps']or q['Torch_imported']:q.update(passed=False,unexpected_model_import=True)
        save();print(json.dumps(dict(path=str(path),SHA256=sha(path),actual_CPU=q['actual_CPU'],passed=q['passed'],elapsed_s=q['elapsed_s'],error=q.get('error'))),flush=True)
    return 0 if q['passed']else 1

if __name__=='__main__':raise SystemExit(main())
