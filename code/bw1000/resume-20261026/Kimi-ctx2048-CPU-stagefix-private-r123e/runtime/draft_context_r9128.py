class ContextStore:
 def __init__(self,max_requests=32):self.epoch=0;self.values={};self.max_requests=max_requests;self.retired=set()
 def begin(self,epoch):
  assert not self.values and type(epoch)is int and epoch>self.epoch;self.epoch=epoch;self.retired=set()
 def validate_many(self,epoch,items):
  assert epoch==self.epoch and 1<=len(items)<=self.max_requests and len({q['request']for q in items})==len(items)
  for q in items:
   key=(epoch,q['request']);assert type(q['request'])is int and 1<=q['request']<=self.max_requests and key not in self.retired;old=self.values.get(key,(0,None))[0];assert type(q['past'])is int and q['past']==old and type(q['rows'])is int and 1<=q['rows']<=1024 and old+q['rows']<=2048
 def append_many(self,epoch,items,parts,concat):
  self.validate_many(epoch,items);assert len(parts)==len(items);staged={}
  for q,part in zip(items,parts):
   key=(epoch,q['request']);old,value=self.values.get(key,(0,None));staged[key]=(old+q['rows'],part if value is None else concat(value,part))
  self.values.update(staged);return [staged[(epoch,q['request'])][1]for q in items]
 def append(self,epoch,request,past,rows,part,concat):return self.append_many(epoch,[{'request':request,'past':past,'rows':rows}],[part],concat)[0]
 def retire(self,epoch,request):
  assert epoch==self.epoch and 1<=request<=self.max_requests;key=(epoch,request);assert key not in self.retired;self.retired.add(key);return self.values.pop(key,None)is not None
 def close(self):keys=list(self.values);self.values.clear();return keys
