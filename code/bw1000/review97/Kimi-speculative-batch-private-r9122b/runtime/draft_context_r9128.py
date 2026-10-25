class ContextStore:
 def __init__(self,max_requests=32):self.epoch=0;self.values={};self.max_requests=max_requests
 def begin(self,epoch):
  assert not self.values and type(epoch)is int and epoch>self.epoch;self.epoch=epoch
 def append(self,epoch,request,past,rows,part,concat):
  assert epoch==self.epoch and type(request)is int and 1<=request<=self.max_requests and 1<=rows<=512
  key=(epoch,request);old,value=self.values.get(key,(0,None));assert old==past and old+rows<=1024
  joined=part if value is None else concat(value,part);self.values[key]=(old+rows,joined);return joined
 def retire(self,epoch,request):
  assert epoch==self.epoch and 1<=request<=self.max_requests;return self.values.pop((epoch,request),None)is not None
 def close(self):keys=list(self.values);self.values.clear();return keys
