"""Read only the original failed862 archive; retain failure and cleanup scopes."""
from pathlib import Path
import argparse,ast,base64,gzip,hashlib,json,re

sha=lambda b:hashlib.sha256(b).hexdigest()
POLICIES=['owned_prefix_policy_r1669','feature_bypass_r1406','indexer_policy_r998','decode_leases_policy_r1356']

def analyze(path,repo):
    blob=Path(path).read_bytes()
    assert sha(blob)=='5a9a8d2677ba9692ea10d3d2cde7114ba1f81fd662feb7f2d5c2c2489f388eb6'
    archive=json.loads(gzip.decompress(blob));files={}
    for entry in archive['files'].values():
        raw=base64.b64decode(entry['base64'])
        assert len(raw)==entry['bytes']and sha(raw)==entry['SHA256']
        assert entry['path']not in files;files[entry['path']]=raw
    assert len(files)==205
    watch=json.loads(files[archive['original_watch']])
    assert watch['completed']and not watch['passed']and watch['actual_GPU']=='910376.862'
    assert watch['CPU_keeper']=='910407.857'and watch['actual_client']is None
    assert not watch['retry']and not watch['reload']and watch['forced_cleanup']
    assert watch['provider_source_SHA256']=='122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'
    def raw_suffix(suffix):
        matches=[v for p,v in files.items()if p.endswith(suffix)]
        assert len(matches)==1,(suffix,len(matches))
        return matches[0]
    source=ast.parse(raw_suffix('/public_glm_native/checkpoint_loader.py'))
    declare=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='declare')
    missing=[]
    for policy in POLICIES:
        name='modern_glm_model_'+policy
        assignments=[t for n in ast.walk(declare)if isinstance(n,ast.Assign)for t in n.targets
            if isinstance(t,ast.Attribute)and t.attr=='argtypes'and isinstance(t.value,ast.Attribute)and t.value.attr==name]
        assert not assignments
        missing.append(name)
    # The successful original418 explicitly set these before entering the same
    # native policy loop. AST only: importing its module would load real Native.
    original=Path(repo)/'code/bw1000/GLM-NoV-W4-quality-r418/execution-v2/owner/model_entry_r290.py'
    tree=ast.parse(original.read_bytes())
    proof={}
    for name in missing:
        assignments=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(
            isinstance(t,ast.Attribute)and t.attr=='argtypes'and isinstance(t.value,ast.Attribute)and t.value.attr==name for t in n.targets)]
        assert len(assignments)==1
        rhs=assignments[0].value
        assert isinstance(rhs,ast.List)and [v.attr for v in rhs.elts]==['c_void_p','c_int']
        proof[name]=dict(original_line=assignments[0].lineno,argtypes=['ctypes.c_void_p','ctypes.c_int'])
    logs=[(p,b.decode(errors='replace'))for p,b in files.items()if p.endswith(('-0.log','-1.log'))]
    assert len(logs)==2
    log_proof=[];full_ready=[]
    for p,text in logs:
        ready=text.count('GLM78_WEIGHTS_READY');crashes=text.count('modern_glm_model_owned_prefix_policy_r1669+0x1a')
        assert ready==8 and crashes==8
        identified=re.findall(r'GLM78_WEIGHTS_READY +([01]) +([0-7]) +([0-9.]+) +\(([0-9]+), *([0-9]+)\)',text)
        full_ready += [dict(stage=int(s),rank=int(r),load_s=float(t),free_bytes=int(f),total_bytes=int(total))for s,r,t,f,total in identified]
        log_proof.append(dict(path=p,SHA256=sha(files[p]),weight_ready_tag_count=ready,
            fully_parseable_weight_ready_records=len(identified),first_policy_crash_frames=crashes))
    assert len(full_ready)==len({(v['stage'],v['rank'])for v in full_ready})==15
    assert not any(p.endswith('/MODEL_READY.json')or '/ready-'in p for p in files)
    assert not any('/closed-'in p for p in files)
    lifecycle=[]
    for p,b in files.items():
        if '/lifecycle-'in p:
            v=json.loads(b);lifecycle.append(dict(path=p,SHA256=sha(b),completed=v['completed'],normal=v['normal'],destroy_attempts=v['destroy_attempts'],shutdown_attempts=v['shutdown_attempts']))
    assert len(lifecycle)==16 and not any(v['normal']for v in lifecycle)
    post=watch['post_after_failure'];assert len(post)==2
    for p in post:
        assert p['actual_step']=='910376.863'and p['CPU_only']and not p['GPU_initialized']and p['KFD_available']
        assert not p['kfd_pids']and not p['matching_component_processes']and not p['expected_PIDs_still_exist']
        assert p['ports_bindable']=={'43843':True}
    rows=archive['accounting']['rows']
    expected={'910376.861':('COMPLETED','0:0'),'910376.862':('FAILED','11:0'),
              '910376.863':('COMPLETED','0:0'),'910407.857':('FAILED','1:0')}
    assert len(rows)==4
    for line in rows:
        parts=line.split('|');assert tuple(parts[2:4])==expected[parts[0]]
    return dict(state='ORIGINAL862_SIGSEGV_BEFORE_READY_OR_HTTP_DECLARATION_GAP_IDENTIFIED_POST_CLEAN',
        archive_SHA256=sha(blob),original_files_verified=205,original_accounting=rows,
        actual_GPU='910376.862 FAILED11:0/14m52s',keeper='910407.857 FAILED1:0/15m07s',
        post='910376.863 COMPLETED0/bothnodesKFD_PIDempty_portbindable',
        provider_manifest_SHA256=watch['provider_source_SHA256'],
        deployment_manifest_SHA256=watch['deployment_source_SHA256'],original_epoch=watch['original_GPU_start_unix'],
        Model_READY=False,actual_HTTP=0,actual_Native_requests=0,original_failed=True,
        weight_ready_tag_count=16,fully_parseable_weight_ready_records=full_ready,
        interleaved_log_limit='16readytags, only15complete stage/rank tuples independently parsed; do not substitute a complete16owner weight receipt.',
        logs=log_proof,all16_normal_destroy_proven=False,lifecycles=lifecycle,
        missing_original_pointer_declarations=proof,
        diagnosis='Static loader extraction declared base create/load/forward API but omitted these four pointer policy signatures from the successful original418. Actual loop passes Python ctx integer; default ctypes integer conversion is unsafe for64bit pointers. Every stage crashes in its first owned_prefix setter. No GPUkernel/quality result was measured.',
        direct_actual_argument_bytes_captured=False,
        next='Originalowner builds immutable changed-source declaration closure and true high64bit ctypes/libffi CPU consumer, preserving originalFAIL. Source/claimpublish and freshidle before one bounded changed-source2request attempt; no blind862replay, no Nativecompile.',
        private_raw_archive_only_L_R=True,performance_round_increment=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();r=analyze(a.archive,a.repo)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:r[k]for k in ['state','original_files_verified','actual_HTTP','weight_ready_tag_count','all16_normal_destroy_proven']},ensure_ascii=False))
