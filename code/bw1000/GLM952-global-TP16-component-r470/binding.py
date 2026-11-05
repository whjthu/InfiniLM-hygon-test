"""Only explicit new component ABI. Import is stdlib-only; never dlopen here."""
import ctypes,json

def declare(lib):
 p=ctypes.c_void_p;i=ctypes.c_int;n=ctypes.c_size_t;c=ctypes.c_char_p
 for name,args,result in [
  ('glm_global_tp16_error_r470',[],c),('glm_global_tp16_id_size_r470',[],n),
  ('glm_global_tp16_id_r470',[p,n],i),
  ('glm_global_tp16_create_r470',[i,i,n,n,c,p,p,n],p),
  ('glm_global_tp16_geometry_r470',[p],c),
  ('glm_global_tp16_probe_r470',[p],i),('glm_global_tp16_close_r470',[p],i)]:
  f=getattr(lib,name);f.argtypes=args;f.restype=result
 return lib

def create(lib,node,device,config,world_id,tp_id):
 assert type(node)is int and node in (0,1)and type(device)is int and 0<=device<8
 assert config.get('glm_global_tp16_r470')is True
 size=lib.glm_global_tp16_id_size_r470();assert 0<size<=4096
 assert type(world_id)is bytes and type(tp_id)is bytes and len(world_id)==len(tp_id)==size and world_id!=tp_id
 buffers=[ctypes.create_string_buffer(x,len(x))for x in (world_id,tp_id)]
 ctx=lib.glm_global_tp16_create_r470(node,device,1,32768,json.dumps(config,separators=(',',':')).encode(),*buffers,size)
 if not ctx:raise RuntimeError(lib.glm_global_tp16_error_r470().decode())
 return ctx # owner thread alone must close; buffer copied synchronously in create
