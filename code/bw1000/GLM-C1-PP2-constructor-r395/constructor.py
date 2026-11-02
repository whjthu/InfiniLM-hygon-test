"""One unchanged public038d Context per owner. Never load/finish/forward.

The same body is consumed by recording CPU providers. Device/provider leaves
remain explicit; a completed constructor does not pass loaded-service admission.
"""
import time

BOUNDS=[0,38,78]
NATIVE_SHA='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'


def body(deps,stage,rank,config,payload,gate,save):
    if stage not in range(2) or rank not in range(8):raise ValueError('Exact2stage/8localowner')
    if config['num_hidden_layers']!=78 or config['hidden_size']!=6144 or config['indexer_types'][BOUNDS[stage]]!='full':
        raise ValueError('Actual78layer config/full-indexer legalcut')
    q=dict(completed=False,stage=stage,rank=rank,global_rank=stage*8+rank,TP=8,PP=2,rows=1,capacity=32768,
           weights_IO=False,finish_calls=0,forward_calls=0,Model_qualified=False,
           normal_destroy=False,normal_shutdown=False,loaded_service_admission_passed=None,
           loaded_service_original_free_floor_bytes=8<<30,Native_SHA256=NATIVE_SHA)
    ctx=None;error=None
    try:
        gate('before-create');q['before_create']=deps.snapshot();save(q)
        if q['before_create']['free_bytes']<=58<<30:raise RuntimeError('Original beforecreate58GiB guard')
        started=time.monotonic();ctx=deps.create(rank,stage,1,32768,config,payload)
        if not ctx:raise RuntimeError('Native constructor returned null')
        q['create_seconds']=time.monotonic()-started
        # StoreContext before any further deadline check so cleanup owns it.
        q['after_create']=deps.snapshot();save(q);gate('after-create')
        part=deps.partition(ctx);keys=part.pop('keys')
        if [part['PP'],part['stage'],part['device'],part['begin'],part['end'],part['weights_ready']]!=[2,stage,rank,BOUNDS[stage],BOUNDS[stage+1],False]:
            raise RuntimeError('Actualpublicconstructorpartition mismatch')
        q['actual_partition']=part;q['parameter_key_count']=len(keys)
        c=deps.census(ctx)
        q['tensor_census']={k:c[k] for k in ['layer_begin','layer_end','rows','capacity','loaded','cache_category_bytes','cache_unique_start_bytes','weight_unique_start_bytes','weight_keys','scope']}
        if [c['layer_begin'],c['layer_end'],c['rows'],c['capacity'],c['loaded']]!=[BOUNDS[stage],BOUNDS[stage+1],1,32768,False]:
            raise RuntimeError('C1 full32K tensor census differs')
        if sum(c['cache_category_bytes'].values())!=c['cache_unique_start_bytes']:raise RuntimeError('KV category accounting')
        gate('before-trim');q['unloaded_trim']=deps.trim(ctx)
        if q['unloaded_trim'][3:]!=[rank,stage,2]:raise RuntimeError('Unloaded trim owner/topology differs')
        # Inherited795 tiny WORLD SUM is solely communicator identity, not Model.
        gate('before-world16-identity');q['world16_identity_sum']=deps.identity_sum(ctx)
        if q['world16_identity_sum']!=16:raise RuntimeError('Actual WORLD must contain16owners')
        q['after_trim']=deps.snapshot()
        q['owner_affinity']=deps.affinity(rank)
        w=c['weight_unique_start_bytes'];kv=c['cache_unique_start_bytes'];s=q['after_trim']
        q['memory_accounting']=dict(weight_tensor_logical_nbytes=w,KV_tensor_logical_nbytes=kv,
            driver_used_increase_bytes=q['before_create']['free_bytes']-s['free_bytes'],
            driver_increment_minus_tensor_logical_bytes=q['before_create']['free_bytes']-s['free_bytes']-w-kv,
            Native_allocator_live_bytes=None,Native_allocator_reserved_bytes=None,
            residual_scope='Mixed Native pool/reservation/CCL/RoPE/otherlive/rounding; not a live-vs-reserved decomposition or a peak.')
        q['would_pass_original_loaded_service_8GiB_guard']=s['free_bytes']>8<<30
        q['loaded_service_admission_passed']=False
        q['loaded_service_admission_scope']='Probe never loads; reporting the original threshold does not authorize lowering/bypassing it.'
        q['minimum_constructor_512MiB_remaining']=s['free_bytes']>=512<<20
        gate('before-normal-destroy')
    except BaseException as e:
        error=e;q.update(error_type=type(e).__name__,error=str(e));save(q)
    finally:
        # Destroy/shutdown are never blocked by an expired/cancelled admissiongate.
        if ctx is not None:
            try:deps.destroy(ctx);q['normal_destroy']=True
            except BaseException as e:q['destroy_error']=repr(e);error=error or e
        try:deps.shutdown();q['normal_shutdown']=True
        except BaseException as e:q['shutdown_error']=repr(e);error=error or e
        try:q['after_shutdown']=deps.snapshot()
        except BaseException as e:q['post_snapshot_error']=repr(e);error=error or e
        q['completed']=error is None and q['normal_destroy'] and q['normal_shutdown']
        save(q)
    if error is not None:raise error
    return q
