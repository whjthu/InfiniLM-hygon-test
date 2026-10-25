import ctypes as C,pathlib,json,hashlib,os,torch,traceback
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();b=json.loads((r/'candidates/Kimi-taps-fixture-r9023/CPU-build.json').read_text());assert b['completed'] and sha(b['native'])==b['native_SHA256'] and sha(b['parent_native'])==b['parent_native_SHA256']
torch.set_num_threads(1);torch.cuda.set_device(0);out=r/'results'/('Kimi-taps-fixture-r9023-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not out.exists();directory=r/'results'/('Kimi-taps-fixture-private-r9023-'+os.environ['SLURM_STEP_ID']);assert not directory.exists();os.environ['INFINILM_KIMI_TARGET_TAPS_DIR_R9022']=str(directory);os.environ.pop('INFINILM_KIMI_TRACE_DIR',None);os.environ['SLURM_PROCID']='0'
lib=C.CDLL(b['native']);target=C.CDLL(b['parent_native']);P=C.c_void_p;lib.taps_fixture_create_r9023.restype=P;lib.taps_fixture_call_r9023.argtypes=[P,C.POINTER(P),C.c_int];lib.taps_fixture_close_r9023.argtypes=[P];lib.taps_fixture_error_r9023.restype=C.c_char_p
target.modern_kimi_diag_arm_r9008.argtypes=[P,C.c_char_p,C.c_int];target.modern_kimi_diag_read_r9008.argtypes=[P];target.modern_kimi_diag_read_r9008.restype=C.c_char_p;target.modern_kimi_diag_capabilities_r9008.restype=C.c_char_p
q={'completed':False,'fixture_only':True,'Model':False,'source_SHA256':sha(__file__),'native':b,'normal_close':False};ctx=None
try:
 ctx=lib.taps_fixture_create_r9023();assert ctx,lib.taps_fixture_error_r9023();assert target.modern_kimi_diag_arm_r9008(ctx,b'fixture',1)==0
 q['actual_capability']=json.loads(target.modern_kimi_diag_capabilities_r9008());hidden=(torch.arange(8*7168,device='cuda',dtype=torch.float32)%97/32).to(torch.bfloat16).reshape(1,8,7168);ssm=torch.zeros((17,6,128,128),device='cuda',dtype=torch.bfloat16);conv=torch.zeros((17,2304,3),device='cuda',dtype=torch.bfloat16);ssm[9].fill_(2);conv[9].fill_(3)
 values=[hidden,ssm,conv,*[torch.tensor(x,device='cuda',dtype=torch.int32)for x in [[536],[544],[0,8],[9],[1]]]];ptrs=(P*8)(*[x.data_ptr()for x in values]);torch.cuda.synchronize()
 assert lib.taps_fixture_call_r9023(ctx,ptrs,0)==0,lib.taps_fixture_error_r9023();before=json.loads(target.modern_kimi_diag_read_r9008(ctx));ssm[1].fill_(5);conv[1].fill_(7);torch.cuda.synchronize();assert lib.taps_fixture_call_r9023(ctx,ptrs,1)==0,lib.taps_fixture_error_r9023();after=json.loads(target.modern_kimi_diag_read_r9008(ctx));records=after['records'];assert after['finite'] and after['actual_callback_invocations']==9 and len(records)==9
 a=before['records'][0];z=next(x for x in records if x['point']=='attn_output');assert a['positions']==z['positions']==[536,537] and a['actual_init_state_indices']==[9] and a['actual_final_state_indices']==[1] and a['past']==[536] and a['total']==[544] and a['offsets']==[0,8]
 for kind in ['ssm','conv']:
  assert a['recurrent_rows_at_boundary'][kind+'_read']==z['recurrent_rows_at_boundary'][kind+'_read'];assert a['recurrent_rows_at_boundary'][kind+'_write']['fnv1a64']!=z['recurrent_rows_at_boundary'][kind+'_write']['fnv1a64']
 expected=hidden.cpu().view(torch.uint16).numpy().tobytes();taps=[x for x in records if x['point']=='decoder'];assert [x['layer']for x in taps]==[19,37,66,78,90]
 for x in taps:assert x['shape']==[1,8,7168] and x['positions']==list(range(536,544)) and pathlib.Path(x['remote_owned_buffer']).read_bytes()==expected
 from target_taps_r9022 import Taps
 features,positions=Taps.assemble([after]);assert list(features.shape)==[1,8,35840] and positions==list(range(536,544));assert all(torch.equal(features[:,:,i*7168:(i+1)*7168],hidden.cpu())for i in range(5))
 for x in records:
  if x['point']!='decoder':assert pathlib.Path(x['remote_owned_buffer']).read_bytes()==expected[:2*7168*2]
 q.update(completed=True,record_count=len(records),callback_count=after['actual_callback_invocations'],read_unchanged_write_changed=True,actual_metadata_pass=True,tap_bytes_pass=True,assembled_shape=list(features.shape),trace_summary=after,raw_activations_stay_remote=True)
finally:
 if ctx:assert target.modern_kimi_diag_arm_r9008(ctx,b'close',0)==0;assert lib.taps_fixture_close_r9023(ctx)==0,lib.taps_fixture_error_r9023();q['normal_close']=True
 q['audit_error']=0 if q['completed'] and q['normal_close']else 1;out.write_text(json.dumps(q,indent=2));print(json.dumps({'completed':q['completed'],'normal_close':q['normal_close'],'audit_error':q['audit_error'],'receipt':str(out)}))
