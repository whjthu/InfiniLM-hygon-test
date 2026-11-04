"""Offline stdlib audit of existing r459 terminal bytes; never executes Native."""
from pathlib import Path
import base64,gzip,hashlib,json,re,sys
H=lambda b:hashlib.sha256(b).hexdigest()
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
P=ROOT+'/incoming/Kimi-AR-TP32-constructor-r459';D=P+'/deployment-v2'
NATIVE='0d4eb39aa2f566876b4defa90747188839724c99ac4db02e8735749956b35eaf'
MF='73ea16bece1ba8d4c42b7aec5538503ba2253208ddeb317cf3fbd16ee6dad21a'

def main(path):
    t=json.loads(Path(path).read_bytes());z=base64.b64decode(t['gzip_base64'])
    assert len(z)==9577069 and H(z)==t['archive_SHA256']=='475396cc6a0408995fa4d6508049798618969c32366bcc1858fb5373d4c3c91f'
    raw=gzip.decompress(z);assert len(raw)==185412412 and H(raw)==t['uncompressed_SHA256']=='335472ee5d04436d41dfd34eba3c3d4b57696f4b35aea1dac3e097ccd63e5203'
    a=json.loads(raw);fs=a['files'];assert len(fs)==63
    for p,v in fs.items():
        b=v['rawtext'].encode();assert len(b)==v['bytes']and H(b)==v['SHA256'],p
    def j(p):return json.loads(fs[p]['rawtext'])
    for root in [P,D]:
        m=j(root+'/SOURCE_MANIFEST.json')
        for n,h in m['files_SHA256'].items():assert fs[root+'/'+n]['SHA256']==h
    m=j(D+'/SOURCE_MANIFEST.json');assert fs[D+'/SOURCE_MANIFEST.json']['SHA256']==MF
    assert m['GPU_GO']and not m['Model_GO']
    c=j(D+'/config.json');assert c['Native_SHA256']==NATIVE and c['floor_bytes']==4294967296
    assert c['environment']['INFINILM_KIMI_MOE_WORKSPACE_R4665']=='0'
    claim=m['resource_claim'];assert (claim['TP'],claim['PP'],claim['cards'],claim['GPU_max_s'],claim['Native_max_s'],claim['CPU_keeper']['max_s'])==(32,1,32,600,540,900)
    for name in ['constructor.py','check_local.py']:
        assert fs[D+'/'+name]['SHA256']==fs[P+'/'+name]['SHA256']
    w=j(ROOT+'/results/Kimi-r459-AR-constructor-ar-tp32-ctor-20261104-b-watch.json')
    assert w['completed']and not w['passed']and w['forced_cleanup']
    assert (w['actual_GPU'],w['CPU_keeper'],w['actual_client'])==('910380.589','910407.972',None)
    assert w['epoch']==1793769510.7776873
    assert (w['Native_deadline'],w['GPU_deadline'],w['keeper_deadline'])==(w['epoch']+540,w['epoch']+600,w['epoch']+900)
    fresh=w['fresh_before'];assert len(fresh)==4 and len({(v['node'],p)for v in fresh for p in v['physical_PCI']})==32
    floor=4294967296;total=68702699520;rows=[];head_rows=[];ranks=[];key_hashes=set()
    for n,node in enumerate(c['nodes']):
        p=ROOT+'/results/Kimi-AR-TP32-constructor-r459-ar-tp32-ctor-20261104-b/node%d.json'%n
        v=j(p);assert v['completed']and not v['passed']and v['error']=="AssertionError('original 4GiB constructor floor')"
        assert (v['run'],v['actual_step'],v['source_manifest_SHA256'])==('ar-tp32-ctor-20261104-b','910380.589',MF)
        assert v['create_calls']==v['destroy_calls']==1 and all(v[k]==0 for k in ['weights','forward','HTTP','reset'])
        assert len(v['entry_mask'])>=32
        ds=[e for e in v['events']if e['event']=='destroy_returned'];assert len(ds)==1 and ds[0]['payload']=={'code':0,'node':n}
        assert not any(e['event']=='cleanup_error'for e in v['events'])
        obs=next(e['payload']for e in v['events']if e['event']=='constructor_observation')
        cap=obs['capabilities'];assert cap['MLA_shape']==[2,132,256,3,192]
        assert all(cap[k]==value for k,value in dict(TP=32,PP=1,logical_capacity=1,physical_rows=1,state_pool_rows=2,context=32768,num_blocks=132,W2_exact_parameter=True,MLA_exact_parameter=True,capture=False,replay=False,GPU_qualified=False).items())
        assert obs['partition']==[0,93]and obs['layout']['TP_node_rank']==n
        assert obs['cache_geometry']['max_rows']==1 and obs['cache_geometry']['pages']==132 and obs['cache_geometry']['KDA_pool_rows']==2
        assert len(obs['keys'])==len(set(obs['keys']))==497052
        actual_layers=sorted({int(match.group(1))for key in obs['keys']if(match:=re.search(r'[.]layers[.](\d+)[.]',key))})
        assert actual_layers==list(range(93))
        contract=obs['key_contract'];assert contract['checkpoint_membership_validated']and contract['actual_layer_ids']==actual_layers
        # Independent set/coverage checks above; checkpoint membership itself was
        # executed by the pinned producer against its actual SHA-bound index.
        key_hashes.add(contract['keys_SHA256'])
        assert obs['head']['expected_CPU_source_registered']is False and len(obs['head']['owners'])==8
        for owner in obs['head']['owners']:
            rank=n*8+owner['device'];ranks.append(rank)
            assert owner['tp_rank']==rank and owner['TP']==32 and owner['PP']==1 and owner['PP_stage']==0
            assert owner['model_constructed']and owner['head_constructed']and owner['physical_head_shape']==[5120,7168]
            assert owner['head_parameter_bytes']==73400320 and not owner['full_GPU_head_retained']and not owner['postload_rows_verified']
            assert owner['head_forward_calls']==owner['head_forward_rows']==0
            assert owner['global_row_begin']==rank*5120 and owner['global_row_end_exclusive']==(rank+1)*5120
            head_rows.append([owner['global_row_begin'],owner['global_row_end_exclusive']])
        assert all(o['changes']==0 and o['verified_mask']==''for o in obs['affinity']['owners']), 'raw constructor snapshot; no lazy submit qualification'
        assert all(value==0 for value in obs['output_metrics'].values())
        for pin in v['dladdr'].values():assert pin['SHA256']==NATIVE and pin['path'].endswith('/libKimi_AR_persistent_storage_r456.so')
        assert {o['PCI']for o in v['hardware']}==set(next(f for f in fresh if f['node']==node)['physical_PCI'])
        for d,((before,tb),(after,ta))in enumerate(zip(v['memory_before_create'],v['memory_after_create'])):
            assert tb==ta==total and before==68365058048 and after<floor
            rows.append(dict(node=node,device=d,tp_rank=n*8+d,beforecreate_free=before,aftercreate_free=after,total=total,
                occupied_before=total-before,occupied_after=total-after,create_driver_delta=before-after,floor_deficit=floor-after))
    assert sorted(ranks)==list(range(32))and len(key_hashes)==1
    post=j(ROOT+'/results/Kimi459-v2-independent-post-first/result.json');assert post['passed']and post['returncode']==0
    ps=[json.loads(line)for line in post['stdout'].splitlines()if line.startswith('{')];assert len(ps)==4
    assert {v['node']for v in ps}==set(c['nodes'])
    for v in ps:
        assert v['actual_step']=='910380.591' and not v['kfd_pids']and not v['matching']and not v['expected_PIDs_still_exist']
        assert all(v['ports_bindable'].values())and v['TCP_connections']==0
    immediate=w['CPU_idle_observations'][-1];assert immediate['phase']=='failure-post'and immediate['returncode']==1
    immediate_rows=[json.loads(line)for line in immediate['stdout'].splitlines()if line.startswith('{')]
    assert any(v['node']=='a14r2n08'and v['ports_bindable']['19743']is False for v in immediate_rows)
    account=fs[ROOT+'/results/Kimi459-589-terminal-accounting-original.txt']['rawtext']
    for step,state,rc in [('910380.588','COMPLETED','0:0'),('910380.589','FAILED','1:0'),('910380.590','FAILED','1:0'),('910380.591','COMPLETED','0:0'),('910407.972','FAILED','1:0')]:
        row=next(v.split('|')for v in account.splitlines()if v.startswith(step+'|'));assert row[2:4]==[state,rc]
    native_logs=[v['rawtext']for p,v in fs.items()if re.search(r'-910380[.]589-[0-3][.]log$',p)]
    assert len(native_logs)==4 and all(s.count('Destroy COMPLETE')==8 for s in native_logs)
    source=j(D+'/WORKSPACE_CONTRACT.json');scenario=source['new_scenario_subtotal']
    assert scenario==64055094076
    free=[v['aftercreate_free']for v in rows];deltas=[v['create_driver_delta']for v in rows]
    return dict(schema=1,audit_passed=True,experiment_passed=False,decision='TP32PP1_AR_STORAGE_CONSTRUCTS_BUT_ALL32_FAIL_UNCHANGED_4GiB_FLOOR_STOP_NO_MODEL',
        GPU='910380.589',keeper='910407.972',independent_post='910380.591',source_manifest_SHA256=MF,Native_SHA256=NATIVE,
        archive=dict(path=t['sealed_path'],SHA256=H(z),bytes=len(z),rawfiles=len(fs)),epoch=w['epoch'],weights=0,forward=0,HTTP=0,
        rank_memory=rows,min_aftercreate_free=min(free),max_aftercreate_free=max(free),floor_deficit_min=floor-max(free),floor_deficit_max=floor-min(free),
        beforecreate_occupied=total-68365058048,create_driver_delta_min=min(deltas),create_driver_delta_max=max(deltas),
        source_classification=dict(parameter_logical_conservative_scenario=62163168060,MLA_exact_logical_pool=1868562432,KDA_regular_pool=20348928,small_scratch_regular_pool=3014656,
            scenario_subtotal=scenario,scenario_kind='Not a complete physical upper/lower bound or HBM traffic',
            driver_create_delta_minus_scenario_min=min(deltas)-scenario,driver_create_delta_minus_scenario_max=max(deltas)-scenario,
            residual_NOT_attributed='Includes unclosed backend allocation rounding/pool/residency, CCL and classification differences; no allocator/CCL dynamic byte census, cannot name all residual as CCL or removable waste'),
        constructor_observations=dict(nodes=4,head_owners=32,head_global_rows=[0,163840],head_bytes_per_rank=73400320,keys_each=497052,layer_range=[0,93],ctx_destroy_returns=[0]*4,RCCL_Destroy_COMPLETE_lines=32,
            raw_affinity_changes=0,lazy_first_submit_binding='NOT_EXERCISED; entry>=32 verified, no fake owner-mask qualification',caps_vs_allocations='Caps/geometry and CPU966 source identify constructor path; no per-bank dynamic allocator census'),
        cleanup=dict(original_GPU_and_keeper_failed=True,TERM_cleanup_record_preserved=w['owned_cleanup'],immediate_post590_failed_head19743=True,independent_post591_clean=True),
        excluded=['GPU OOM assertion','Model fit','performance','forward H3/I96 numeric qualification','full benchmark','automatic next GPU'],GPU_GO=False,Model_GO=False)

if __name__=='__main__':print(json.dumps(main(sys.argv[1]),indent=2)+'\n',end='')
