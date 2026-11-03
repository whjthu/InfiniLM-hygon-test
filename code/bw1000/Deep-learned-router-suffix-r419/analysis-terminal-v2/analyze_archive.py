"""Independent stdlib reader of original GPU1351 raw proofs and same-query timing."""
from pathlib import Path
import argparse,ast,base64,hashlib,io,json,math,statistics,struct,zipfile
sha=lambda b:hashlib.sha256(b).hexdigest()
F=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
MODES=[0,1,0,1,1,0,1,0,0,1]
PAIRS=[(2,3),(5,4),(7,6),(8,9)]  # (OFF, ON), preserve matching query index/owner.
NATIVE='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
BRIDGE='17153310c14d065a877ea047394bed049eb619c24b4f755fc4655fa01aa66348'
REPORT='419eb4aa50b12d608e262f802ec8577d2353edecead20e30539926f5e26f057d'

def norm(a,b):
    return math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))/(math.sqrt(math.fsum(y*y for y in b))+1e-30)

def error(a,b):
    return {'maxabs':max(abs(x-y)for x,y in zip(a,b)), 'NRMSE':norm(a,b)}

def npz_arrays(payload):
    out={}
    with zipfile.ZipFile(io.BytesIO(payload))as z:
        for name in z.namelist():
            assert name.endswith('.npy')and '/'not in name
            b=z.read(name);assert b[:6]==b'\x93NUMPY';version=b[6:8]
            if version==b'\x01\x00':length=struct.unpack('<H',b[8:10])[0];start=10
            else:assert version in (b'\x02\x00',b'\x03\x00');length=struct.unpack('<I',b[8:12])[0];start=12
            assert length<4096;h=ast.literal_eval(b[start:start+length].decode().strip());raw=b[start+length:]
            assert h['descr']in ('|u1','<u1')and h['fortran_order']is False and h['shape']==(len(raw),)
            out[name[:-4]]=raw
    return out

def key(p):
    return p.split('/results/',1)[1]

def read(archive):
    original=Path(archive).read_bytes();q=json.loads(original);assert q['schema']=='Deep419-terminal-proof-archive-v1';files={}
    for item in q['files']:
        b=base64.b64decode(item['base64'])if'base64'in item else item['text'].encode()
        assert len(b)==item['bytes']and sha(b)==item['SHA256'];k=key(item['path']);assert k not in files;files[k]=b
    assert len(files)==228
    return q,files,sha(original)

def decode(raw,meta,kind,shape):
    assert meta['shape']==shape and meta['dtype']=={'BF16':9,'F32':10,'I64':3}[kind]
    size={'BF16':2,'F32':4,'I64':8}[kind];assert len(raw)==math.prod(shape)*size and sha(raw)==meta['SHA256']
    if kind=='BF16':out=[struct.unpack('<f',struct.pack('<I',x<<16))[0]for(x,)in struct.iter_unpack('<H',raw)]
    else:out=[x for(x,)in struct.iter_unpack('<f'if kind=='F32'else'<q',raw)]
    assert all(math.isfinite(v)for v in out);return out

def softplus_sqrt(x):
    soft=x if x>20 else F(math.log1p(F(math.exp(x))))
    return F(math.sqrt(soft))

def normalize(scores,ids):
    v=[scores[i]for i in ids];total=F(F(F(v[0]+v[1])+F(v[2]+v[3]))+F(v[4]+v[5]));denom=F(total+F(1e-20))
    return [F(F(x/denom)*F(1.5))for x in v]

def suffix_check(packet,reference=None):
    scores,selection,ids,weights=packet['scores'],packet['selection'],packet['ids'],packet['weights']
    assert len(set(ids))==6 and all(0<=v<256 for v in ids)
    assert [selection[i]for i in ids]==sorted(selection,reverse=True)[:6]
    own=error(weights,normalize(scores,ids));assert own['maxabs']<2e-6 and own['NRMSE']<2e-6
    if reference is None:return {'own_score_F32_tree_normalization':own}
    e=error(scores,reference['scores']);bound=max(2e-6,5e-7*max(abs(x)for x in reference['scores']));assert e['maxabs']<=bound
    se=error(selection,reference['selection']);assert se['maxabs']<=max(2e-6,5e-7*max(abs(x)for x in reference['selection']))
    ordered=sorted(reference['selection'],reverse=True);margin=ordered[5]-ordered[6];separated=margin>2*se['maxabs'];same=set(ids)==set(reference['ids'])
    if separated:assert same,'Margin-separated wrong expert set'
    derived=error(weights,normalize(reference['scores'],ids));assert derived['maxabs']<2e-6 and derived['NRMSE']<2e-6
    return dict(own_score_F32_tree_normalization=own,score=e,selection=se,cutoff_margin=margin,margin_separated=separated,same_expert_set=same,literal_route_order_equal=ids==reference['ids'],literal_weights=error(weights,reference['weights']),reference_unbiased_score_normalization_at_actual_IDs=derived)

def metric_summary(values):
    assert len(values)==10
    ratios=[values[on]/values[off]for off,on in PAIRS]
    block_ratios=[((values[3]+values[4])/2)/((values[2]+values[5])/2),((values[6]+values[9])/2)/((values[7]+values[8])/2)]
    off=statistics.mean(values[i]for i in [2,5,7,8]);on=statistics.mean(values[i]for i in [3,4,6,9])
    return dict(same_query_max8_s=values,formal_OFF_mean_s=off,formal_ON_mean_s=on,ON_over_OFF=on/off,improvement_percent=100*(1-on/off),adjacent_pairs=[dict(OFF_index=o,ON_index=n,ON_over_OFF=r)for(o,n),r in zip(PAIRS,ratios)],pairs_faster=sum(r<1 for r in ratios),block_ON_over_OFF=block_ratios,OFF_endpoint_drift_signed=[values[5]/values[2]-1,values[8]/values[7]-1],cross_block_OFF_change=statistics.mean(values[i]for i in [7,8])/statistics.mean(values[i]for i in [2,5])-1)

def analyze(archive,bias_witness):
    q,files,archive_sha=read(archive);report_bytes=files['Deep419-MoE-910374.1351.json'];assert sha(report_bytes)==REPORT;r=json.loads(report_bytes)
    w=json.loads(files['Deep-learned-router-suffix-r419-v2-watch.json']);assert w['completed']and w['passed']and w['srun_exit_code']==0 and w['actual_GPU']=='910374.1351'
    assert w['CPU_keeper']=='910407.854'and w['normal_all8']and w['post']['actual_step']=='910374.1352'and w['report_SHA256']==REPORT
    for phase in ['fresh_before','post']:
        v=w[phase];assert v['KFD_available']and not v['kfd_pids']and not v['matching_component_processes']and not v['port43882_listening']
    assert len(q['accounting'])==4 and all('|COMPLETED|0:0|'in line for line in q['accounting'])
    assert r['completed']and r['passed']and r['normal']and r['actual_step']=='910374.1351'
    assert r['Native_SHA256']==NATIVE and r['bridge_SHA256']==BRIDGE and not r['Model_loaded']and r['phase_observer']is False
    assert r['actual_SDK_metadata_executed']and r['metadata_startup_returncode']==0 and r['SDK_metadata_used_as_math_gate']is False and r['metadata_exe_static_GPU_queries_not_zero']
    metadata_bytes=files[key(r['metadata_stdout_path'])];assert sha(metadata_bytes)==r['metadata_stdout_SHA256'];assert json.loads(metadata_bytes.decode().splitlines()[-1])==r['SDK_metadata']
    assert len(r['SDK_metadata']['SDK_default_metadata_only'])==9
    bias_raw_file=Path(bias_witness).read_bytes();b=json.loads(bias_raw_file);bias_raw=base64.b64decode(b['base64']);assert len(bias_raw)==b['bytes']==1024 and sha(bias_raw)==b['SHA256']
    assert b['dtype']=='F32'and b['shape']==[256]and b['SHA256']==r['checkpoint']['selected_weight_payload_SHA256']['layers.16.ffn.gate.bias'];bias=list(struct.unpack('<256f',bias_raw))
    assert len(r['owners'])==8 and len(r['raw_archives'])==112 and len(r['hardware'])==8 and len({v['PCI']for v in r['hardware']})==8
    expected={(rank,'query'+str(i))for rank in range(8)for i in range(10)}|{(rank,tag+str(mode))for rank in range(8)for tag in ['same-logits-mode','T2-fallback-mode']for mode in [0,1]};packets={};sidecars={}
    for item in r['raw_archives']:
        pair=(item['rank'],item['tag']);assert pair in expected and pair not in packets
        raw=files[key(item['npz'])];jraw=files[key(item['metadata'])];assert sha(raw)==item['npz_SHA256']and sha(jraw)==item['metadata_SHA256'];side=json.loads(jraw);arrays=npz_arrays(raw);assert set(arrays)==set(side['fields'])
        tag=item['tag'];packet={}
        if tag.startswith('same-logits'):
            spec={1:('weights','F32',[1,6]),2:('ids','I64',[1,6]),3:('logits','F32',[1,256]),4:('scores','F32',[1,256]),5:('selection','F32',[1,256])}
        else:
            tokens=2 if tag.startswith('T2')else 1;spec={0:('value','BF16',[tokens,4096]),1:('weights','F32',[tokens,6]),2:('ids','I64',[tokens,6])}
        assert set(arrays)=={str(k)for k in spec}
        for k,(name,kind,shape)in spec.items():packet[name]=decode(arrays[str(k)],side['fields'][str(k)],kind,shape)
        if 'value'in packet:
            for offset in range(0,len(packet['ids']),6):assert len(set(packet['ids'][offset:offset+6]))==6 and all(0<=x<256 for x in packet['ids'][offset:offset+6])
        packets[pair]=packet;sidecars[pair]=side
    assert set(packets)==expected
    suffix=[];fallback=[];output_max=0.;weight_max=0.;OFF_hashes=set();ON_hashes=set();all_affinity=[]
    for rank,owner in enumerate(r['owners']):
        assert owner['rank']==rank and owner['loaded_key_count']==1544 and owner['normal_close']and owner['destroy_returned']and owner['Runtime_expired']and owner['same_logits_qualified']
        assert owner['actual_affinity']==owner['expected_affinity']and len(owner['actual_affinity'])==4;all_affinity+=owner['actual_affinity'];assert len(owner['initial_affinity'])>=32
        sdk=packets[(rank,'same-logits-mode0')];candidate=packets[(rank,'same-logits-mode1')];assert sdk['logits']==candidate['logits']
        assert sidecars[(rank,'same-logits-mode0')]['fields']['3']['SHA256']==sidecars[(rank,'same-logits-mode1')]['fields']['3']['SHA256']
        direct=suffix_check(candidate,sdk);suffix_check(sdk)
        oracle_scores=[softplus_sqrt(x)for x in sdk['logits']];oracle_selection=[F(s+bias[i])for i,s in enumerate(oracle_scores)]
        cpu_score_error=error(sdk['scores'],oracle_scores);assert cpu_score_error['maxabs']<=max(2e-6,5e-7*max(oracle_scores))
        for p in [sdk,candidate]:assert error(p['selection'],[F(s+bias[i])for i,s in enumerate(p['scores'])])['maxabs']<2e-6
        suffix.append(dict(rank=rank,selected_experts=sdk['ids'],SDK_vs_fused=direct,SDK_vs_independent_CPU_libmath_F32_score=cpu_score_error))
        for mode in [0,1]:
            st=sidecars[(rank,'same-logits-mode'+str(mode))]['stats'];assert(st['calls'],st['hits'],st['fallbacks'])==(1,mode,1-mode)
        a=packets[(rank,'T2-fallback-mode0')];c=packets[(rank,'T2-fallback-mode1')];assert a['ids']==c['ids'];e=error(c['value'],a['value']);assert e['NRMSE']<.01
        for mode in [0,1]:
            st=sidecars[(rank,'T2-fallback-mode'+str(mode))]['stats'];assert(st['calls'],st['hits'],st['fallbacks'])==(1,0,1)
        fallback.append(dict(rank=rank,complete_output=e,literal_weights=error(c['weights'],a['weights'])))
        assert len(owner['calls'])==10 and len(owner['same_logit_checks'])==2
        for i,rec in enumerate(owner['calls']):
            assert rec['index']==i and rec['mode']==MODES[i]and rec['phase']==('warm'if i<2 else'formal')
            st=rec['stats'];assert(st['calls'],st['hits'],st['fallbacks'])==(1,MODES[i],1-MODES[i])
            assert st['owner_thread_hash']==owner['calls'][0]['stats']['owner_thread_hash']and rec['proof_D2H_bytes']==8264 and rec['proof_D2H_sync_calls']==3
            p=packets[(rank,'query'+str(i))];ref=packets[(rank,'query0')];e=error(p['value'],ref['value']);assert e['NRMSE']<.01;output_max=max(output_max,e['NRMSE'])
            assert abs(e['NRMSE']-rec['output_vs_first_OFF_NRMSE'])<1e-12 and p['ids']==sdk['ids']
            reference=candidate if MODES[i]else sdk;we=error(p['weights'],reference['weights']);weight_max=max(weight_max,we['maxabs']);assert we['maxabs']<2e-6
            (ON_hashes if MODES[i]else OFF_hashes).add(sha(struct.pack('<'+str(len(p['value']))+'f',*p['value'])))
    assert len(set(all_affinity))==32 and len({o['TID']for o in r['owners']})==8
    journal=[json.loads(x)for x in files['Deep419-MoE-910374.1351/scalar.jsonl'].decode().splitlines()];assert len(journal)==80 and len({(x['rank'],x['index'])for x in journal})==80
    for item in journal:
        original=dict(item);rank=original.pop('rank');assert original==r['owners'][rank]['calls'][item['index']]
    CPP=[];CAPI=[];proof=[];checked=[];critical=[];query_enclosures=[]
    for i in range(10):
        rows=[owner['calls'][i]for owner in r['owners']];cr=max(range(8),key=lambda k:rows[k]['stats']['MoE_complete_s']);CPP.append(rows[cr]['stats']['MoE_complete_s']);critical.append(cr)
        CAPI.append(max(x['forward_CAPI_s']for x in rows));proof.append(max(x['CAPI_with_proof_s']for x in rows));checked.append(max(x['measurement_end_s']for x in rows)-min(x['measurement_begin_s']for x in rows))
        source=r['matched_queries'][i]
        for field,value in [('max8_CPP_complete_s',CPP[-1]),('max8_forward_CAPI_s',CAPI[-1]),('max8_CAPI_with_proof_s',proof[-1]),('same_trial_checked_batch_wall_s',checked[-1])]:assert abs(source[field]-value)<1e-12
        for rank in range(8):
            p=packets[(rank,'query'+str(i))];reference=packets[(0,'query'+str(i))];assert p['ids']==reference['ids'];assert norm(p['value'],reference['value'])<.002
        query_enclosures.append(dict(index=i,mode=MODES[i],CPP_critical_owner=cr,max8_CPP_s=CPP[-1],max8_forward_CAPI_s=CAPI[-1],max8_proof_CAPI_s=proof[-1],checked_batch_s=checked[-1]))
    assert r['CPU_source_weights_and_hidden_canary_unchanged']and r['checkpoint']['selected_key_count']==1544
    for providers in [r['initial_providers'],r['final_providers']]:assert {x['SHA256']for x in providers if'libDeep_'in x['path']}=={NATIVE,BRIDGE}
    paired_output=[]
    for off,on in PAIRS:
        errors=[norm(packets[(rank,'query'+str(on))]['value'],packets[(rank,'query'+str(off))]['value'])for rank in range(8)];paired_output.append(dict(OFF_index=off,ON_index=on,max8_output_NRMSE=max(errors)))
    result=dict(id='Deep419-GPU1351-complete-MoE-independent-audit',reader_passed=True,archive_SHA256=archive_sha,archive_bytes=Path(archive).stat().st_size,report_SHA256=REPORT,bias_witness_SHA256=sha(bias_raw_file),actual_GPU='910374.1351 COMPLETED0/100s',actual_keeper='910407.854 COMPLETED0/112s',post=w['post'],source_manifest_SHA256=r['source_manifest_SHA256'],Native_SHA256=NATIVE,bridge_SHA256=BRIDGE,raw_files_hash_checked=228,raw_NPZ_and_sidecar_pairs=112,numerical=dict(finite_and_actual_dtype_shapes=True,same_logit_pairs=8,suffix_per_owner=suffix,untimed_T2_fallback=fallback,complete_T1_query_count=80,complete_T2_query_count=16,complete_MoE_max_output_NRMSE_vs_first_OFF=output_max,formal_neighbor_output_NRMSE=paired_output,complete_T1_route_IDs_match_same_logit=True,max_complete_weight_difference_from_same_logit=weight_max,OFF_output_distinct_F32_hashes=len(OFF_hashes),ON_output_distinct_F32_hashes=len(ON_hashes),all_source_original_margins_separated=True,CPU_oracle_limit='Python libmath rounded to F32 intermediates, not a claim of bitexact ATen GPU libmath. Direct SDK same-logits and actual complete outputs checked independently.'),timing=dict(CPP_complete=metric_summary(CPP),forward_CAPI=metric_summary(CAPI),CAPI_with_proof=metric_summary(proof),checked_batch=metric_summary(checked),per_query=query_enclosures,boundaries='CPP_complete is host wall enclosing whole MoE and original final stream drain; not GPU event time. Forward CAPI adds external pointer/producer-stream checks. CAPI+proof additionally has three D2H+sync, stats and GPU input canary. Checked batch additionally has NumPy output checks, archive writes and barriers; independent Torch suffix oracle belongs to untimed qualification. Nested scopes must not be added.'),cost=dict(driver_wall_s=r['driver_wall_s'],SDK_device_query_startup_s=r['SDK_device_query_startup_s'],selected_weight_read_hash_s=r['weight_payload_read_and_hash_s'],all_owner_init_qualification_trials_close_s=r['all_owner_initialize_qualify_trial_close_s'],CPP_max8_all_10_query_sum_s=sum(CPP),CPP_max8_formal_8_query_sum_s=sum(CPP[2:]),Slurm_GPU_allocation_s=100,Slurm_keeper_s=112,not_GPU_busy_measurement=True,disjoint_cost_warning='Reported phases are not a complete disjoint wall partition; startup/import/provider/hardware/canary work also exists. Do not sum nested query scopes.'),normal_eight_Runtime_destroy_expired=True,owner_threads_exited=True,proof_canaries=dict(same_logit_bytes_equal_independently=True,CPU_source_weights_and_external_input='Producer asserted every input canary and final CPU payload hashes; source reviewed. No independent before/after GPU-weight byte readback exists.',private_GPU_weight_bytes='UNKNOWN'),actual_SDK_metadata=r['SDK_metadata'],metadata_startup_really_queries_devices=True,original_CPU851_producer_passed=False,CPU_original_validate_PASS=13,CPU_known_backend_underflow_FAIL_classified=1,Model_qualification=False,real_Model_activation_qualification=False,performance_scope='One actual complete layer16 learned MoE on one synthetic immutable BF16 activation, same TP8 Runtime/process. No Model multiplier or full-model gain claim.',decision='Retain limited complete-MoE CPP positive signal if its four pairs exceed within-block CPP OFF drift. Forward CAPI second-block drift is material; proof/checked costs do not show robust gain. No automatic Model reload, grid or repeated card.')
    cpp=result['timing']['CPP_complete'];drift=max(abs(x)for x in cpp['OFF_endpoint_drift_signed'])
    assert cpp['pairs_faster']==4 and all(1-p['ON_over_OFF']>drift for p in cpp['adjacent_pairs'])
    result['decision']='SCOPED_COMPLETE_MOE_CPP_GAIN_RETAINED_NO_MODEL_PROMOTION'
    result['decision_basis']='All four matched CPP pairs improved more than the maximum within-block CPP OFF endpoint drift. Forward CAPI also has four faster pairs but second-block OFF drift is 18.72%; CAPI+proof and checked harness do not support a robust full checked-path speedup. Retain only this layer16/synthetic-activation/TP8 component signal; no new grid or standalone Model reload.'
    result['raw_archive_filename']=Path(archive).name;result['bias_witness_filename']=Path(bias_witness).name
    result['typed_raw_array_count']=sum(len(v['fields'])for v in sidecars.values())
    result['numerical']['all_source_original_margins_separated']=all(v['SDK_vs_fused']['margin_separated']for v in suffix)
    result['cost']['driver_wall_not_partitioned_by_the_three_reported_stage_fields_s']=r['driver_wall_s']-r['SDK_device_query_startup_s']-r['weight_payload_read_and_hash_s']-r['all_owner_initialize_qualify_trial_close_s']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('bias_witness');p.add_argument('--out');a=p.parse_args();result=analyze(a.archive,a.bias_witness);text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.out:Path(a.out).write_text(text,encoding='utf-8')
    else:print(text,end='')
