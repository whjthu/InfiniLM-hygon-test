"""Known CPU-recording node field repair; inherits942 source/Bash proof only."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;P=D.parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
OUT=R/'candidates/Kimi-wide-fewshot-quality-r448-client-CPU-repair-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not OUT.exists();OUT.mkdir();path=OUT/'CPU-result.json';start=time.monotonic()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,SDK=False,weights=False,Model_READY=False,encoding937_repeated=False,prior942_failed_preserved=True,old942_preflight_repeated=False)
    def save():
        t=path.with_suffix('.tmp');t.write_bytes((json.dumps(q,indent=2,allow_nan=False)+'\n').encode());t.replace(path)
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
        mf=sha(D/'SOURCE_MANIFEST.json');assert mf==os.environ['KIMI448_CLIENT_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
        for n,h in m['files_SHA256'].items():assert sha(D/n)==h
        assert sha(P/'SOURCE_MANIFEST.json')==m['frozen_production_manifest_SHA256'];parent=json.loads((P/'SOURCE_MANIFEST.json').read_bytes())
        for n,h in parent['files_SHA256'].items():assert sha(P/n)==h
        prior=R/'candidates/Kimi-wide-fewshot-quality-r448-client-CPU-v1/CPU-result.json';assert sha(prior)==m['prior942_receipt_SHA256'];old=json.loads(prior.read_bytes())
        assert old['completed']and not old['passed']and old['actual_CPU']=='910407.942'and old['error']=="KeyError('node')"
        assert old['phases']==['actual_controller_verify_scope_source937_and_parent_metadata_bytes','actual_final_Bash_exports_typedCLI4roles_noServe']and old['final_Bash_and_typedCLI']['passed']
        assert not old['project_maps']and not old['Torch_imported']
        q.update(source_manifest_SHA256=mf,production_manifest_SHA256=m['frozen_production_manifest_SHA256'],inherited942_source_and_actualBash4role_proof=dict(receipt_SHA256=sha(prior),passed_phases=old['phases']),actual_encoding_CPU_receipt_SHA256=parent['encoding_CPU_receipt_SHA256'])
        def deny_weights(event,args):
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('CPU repair forbids weight payloads')
        sys.addaudithook(deny_weights)
        sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('Kimi448_repaired_recording_consumer',D/'check_entry_CPU.py');consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
        q['actual_changed_entry']=consumer.check(P,parent);assert q['actual_changed_entry']['passed'];q['passed']=True
    except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start,project_maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if any(v in s for v in ['_infinilm.cpython','libKimi_','libDeep_','libGLM_','libinfinilm_runtime','libinfiniops','libinfinirt','libinfiniccl'])],Torch_imported='torch'in sys.modules)
        if q['project_maps']or q['Torch_imported']:q.update(passed=False,unexpected_model_import=True)
        save();print(json.dumps(dict(path=str(path),SHA256=sha(path),actual_CPU=q['actual_CPU'],passed=q['passed'],elapsed_s=q['elapsed_s'],error=q.get('error'))),flush=True)
    return 0 if q['passed']else 1

if __name__=='__main__':raise SystemExit(main())
