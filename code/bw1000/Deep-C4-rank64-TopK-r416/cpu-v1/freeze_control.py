"""Shared actual owner staging consumer: no metadata rollback, no skipped original prefix."""
import hashlib,threading,time
import numpy as np

def signature(raw,state,oracle,dtypes):
 assert raw['dtype']==dtypes['U8'] and tuple(raw['shape'])==(128,8448)
 assert state['dtype']==dtypes['BF16'] and tuple(state['shape'])==(8,512)
 keys,scales=oracle.gather_paged(raw['raw'].reshape(raw['shape']),np.arange(2124),np.ones(2124,np.uint8))
 assert np.isfinite(scales).all() and np.isfinite((state['raw'].view(np.uint16).astype(np.uint32)<<16).view(np.float32)).all()
 return dict(valid_raw_SHA256=hashlib.sha256(keys.tobytes()+scales.tobytes()).hexdigest(),state_SHA256=hashlib.sha256(state['raw'].tobytes()).hexdigest(),state_dtype=state['dtype'],state_shape=tuple(state['shape']))

def stage(f,check,ctx,item,api,oracle,dtypes,control,input_identity):
 tick=time.perf_counter();tid=threading.get_ident()
 assert item['past']==8496 and item['T']==1
 if not control:
  assert item['index']==0
  control.update(thread=tid,input_identity=dict(input_identity))
  check(f('freeze')(ctx))
 else:assert control['thread']==tid and control['input_identity']==input_identity
 check(f('restore')(ctx))
 raw=api.proof(f,check,ctx,6);state=api.proof(f,check,ctx,7)
 sig=signature(raw,state,oracle,dtypes);info=api.freeze_stats(f,check,ctx)
 assert info['captures']==1 and info['restores']==item['index']+1 and info['valid_raw_bytes']==280368 and info['state_bytes']==8192 and info['D2D_copies']==4
 assert info['prefix_generation']==info['restored_generation']==item['index']+1
 if 'signature'not in control:control['signature']=sig
 else:assert sig==control['signature'],'restored effective cache/state differs from first frozen payload'
 if 'last_revision'in control:assert info['realprefix_revision']>control['last_revision'],'real prefix revision not regenerated'
 control['last_revision']=info['realprefix_revision']
 return dict(**info,**sig,input_identity=dict(input_identity),same_frozen_effective_payload=True,metadata_from_real_prefix=True,stage_complete_CAPI_s=time.perf_counter()-tick,freeze_once_this_query=item['index']==0,restore_added_sync_calls=1,freeze_added_sync_calls=1 if item['index']==0 else 0,restore_D2D_bytes=288560,restore_signature_proof_full_raw_D2H_bytes=1081344,restore_signature_state_D2H_bytes=8192,proof_sync_scope='Two existing proof_read calls each Runtime copy drain plus explicit drain; new staging cost is symmetric and included, never free.')
