"""Boundary-only CPU allocator bookkeeping; no GPU sync/trim/finish/native call."""
def snapshot(torch,label):
 rows=[]
 for device in range(8):
  free,total=torch.cuda.mem_get_info(device)
  row={'device':device,'driver_free':int(free),'driver_total':int(total)}
  for name,api in [('torch_allocated','memory_allocated'),('torch_reserved','memory_reserved')]:
   f=getattr(torch.cuda,api,None)
   try:row[name]=int(f(device))if callable(f)else 'NOT_AVAILABLE'
   except (AttributeError,NotImplementedError,RuntimeError,TypeError)as e:row[name]='NOT_AVAILABLE';row[name+'_reason']=type(e).__name__
  row['native_pool_active']=row['native_pool_cached']='NOT_AVAILABLE_NO_SAFE_EXISTING_CAPI'
  rows.append(row)
 return {'label':label,'devices':rows,'boundary_only':True,'no_native_pool_inference':True}
