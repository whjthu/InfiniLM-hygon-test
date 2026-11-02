"""Once-only Linux HOST closure, no library dlopen, GPU or weights."""
from pathlib import Path
import ast,datetime,hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=R/'candidates/GLM-C1-PP2-constructor-r395';out.mkdir(exist_ok=False);start=time.monotonic()
    q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights=False,compile=False,
           actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))
    try:
        assert q['source_manifest_SHA256']==os.environ['GLM_R395_CPU_SOURCE_SHA']
        assert all(os.environ.get(k)=='' for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        assert not os.environ.get('LD_PRELOAD')
        manifest=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
        for name,h in manifest['files_SHA256'].items():assert sha(D/name)==h,name
        for p in D.glob('*.py'):ast.parse(p.read_text())
        from native_chain_r186 import verify
        q['actual038d_ancestry']=verify()
        from public_api import SYMBOLS
        names={line.split()[-1] for line in subprocess.check_output(['nm','-D','--defined-only',q['actual038d_ancestry']['library_actual_path']],text=True,timeout=30).splitlines() if line.split()}
        assert set(SYMBOLS)<=names
        q['actual_symbols_present']=SYMBOLS
        cfg=Path('/old/models/ZhipuAI-GLM-5.3/config.json');c=json.loads(cfg.read_bytes())
        assert c['num_hidden_layers']==78 and c['hidden_size']==6144 and c['indexer_types'][0]==c['indexer_types'][38]=='full' and c['max_position_embeddings']>=32768
        q['config_SHA256']=sha(cfg);q['actual_config_geometry']={k:c[k] for k in ['num_hidden_layers','hidden_size','vocab_size','max_position_embeddings']}
        from check_CPU import run,check_go
        q['actual_adapter_body_CPU']=run()
        q['actual_GPU_entry_GO_contract_CPU']=check_go(q['config_SHA256'])
        tree=ast.parse((D/'public_api.py').read_text()+ '\n'+(D/'constructor.py').read_text())
        forbidden={'modern_glm_model_load_batch','modern_glm_model_finish','modern_glm_model_forward','modern_glm_model_forward_batch'}
        used={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        assert not used&forbidden
        q['forbidden_weight_finish_forward_calls_absent']=True
        q['exact_legacy_GPU_API_source_refs']=manifest['parent_sources']
        assert 'torch'not in sys.modules and 'infinilm'not in sys.modules
        q.update(passed=True)
    except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start)
        (out/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
        print(json.dumps({k:q.get(k) for k in ['completed','passed','actual_CPU','elapsed_s','error']}),flush=True)
    return 0 if q['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
