"""One bounded changed-entry CPU consumer; no encoder/Native/SDK execution."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Deep-a314-fixed-cohort-C4-r458-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    B.mkdir(exist_ok=False);start=time.monotonic()
    q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        Native_dlopen=False,SDK_execution=False,Runtime_execution=False,weights=False,GPU=0,
        Model_READY=False,new_compiles=0,repeated_CPU945_encoding=False,old_suites_replayed=False)
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==os.environ['DEEP458_SOURCE_SHA256']
        q['source_manifest_SHA256']=hashlib.sha256(raw).hexdigest()
        for name,h in json.loads(raw)['files_SHA256'].items():assert sha(D/name)==h,name
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth')):
                raise RuntimeError('CPU does not open weight payload')
            if event=='ctypes.dlopen' and args and args[0] and any(s in str(args[0])for s in ['libDeep_','libGLM_','libKimi_','_infinilm','libinfiniops','libinfinirt','libinfiniccl']):
                raise RuntimeError('CPU forbids Native/SDK dlopen')
        sys.addaudithook(guard)
        from fixed_cohort.entry import prepare
        prepared=prepare(D/'CPU_config.json')
        assert prepared['descriptor']['rows']==4 and prepared['descriptor']['capacity_per_row']==32768
        assert prepared['descriptor']['Model_READY'] is False and not prepared['descriptor']['standard_public_C1']
        private=B/'bound-packet-private.json'
        private.write_text(json.dumps(prepared,ensure_ascii=False)+'\n',encoding='utf-8')
        packet=json.loads(private.read_bytes())
        assert packet['records']==json.loads(json.dumps(prepared['records']))
        q['actual_packet_handoff']=dict(path=str(private),bytes=private.stat().st_size,SHA256=sha(private),
            input_ordinals=[r['ordinal']for r in packet['records']],cases=[dict(case_id=r['case_id'],
                prompt_IDs_SHA256=r['prompt_IDs_SHA256'],body_SHA256=r['body_SHA256'],
                prompt_tokens=len(r['prompt_token_ids']),output_budget=r['max_new_tokens'])for r in packet['records']])
        from check_CPU import checks
        q['changed_consumer']=checks(packet,B/'recording-private')
        assert q['changed_consumer']['passed']
        negative=B/'bad-profile.json';cfg=json.loads((D/'CPU_config.json').read_bytes());cfg['rows']=1;negative.write_text(json.dumps(cfg))
        try:prepare(negative)
        except AssertionError:q['C1_descriptor_rejected_before_binding']=True
        else:raise AssertionError('private C1 profile accepted')
        from fixed_cohort.inputs import load_four
        bad=B/'bad-encoding.json';bad.write_text('{}')
        try:load_four(bad,cfg['gold'])
        except AssertionError:q['foreign_encoding_rejected_before_records']=True
        else:raise AssertionError('foreign input accepted')
        q['actual_a314_metadata_derivation']=dict(library_SHA256=prepared['receipt']['library_SHA256'],
            actual_parent_CPU=prepared['receipt']['actual_step'],objects=len(prepared['normalized']['link']['full']['objects']),
            descriptor_rows=4,capacity_per_row=32768,Native_dlopen=False)
        assert q['actual_a314_metadata_derivation']['objects']==50
        maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if any(t in s for t in ['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfiniops','libinfinilm_runtime','libinfinirt','libinfiniccl'])]
        q['project_or_SDK_maps']=maps;assert not maps
        assert 'torch'not in sys.modules and 'flash_attn'not in sys.modules
        q.update(passed=True,GPU_GO=False,Model_GO=False,actual_four_CPU945_inputs=True,
            full32K_rows4_device_floor='UNKNOWN',recording_leaves=['Native libffi callbacks','checkpoint load/finish','providers/affinity/thread census','GPU memory','BLAS'])
    except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        q.update(completed=True,elapsed_s=time.monotonic()-start)
        (B/'CPU-result.json').write_text(json.dumps(q,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error']}),flush=True)
    return 0 if q['passed']else 1

if __name__=='__main__':raise SystemExit(main())
