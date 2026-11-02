"""Append one scalar record/update per callback; aggregate only at terminal."""
from pathlib import Path
import json
class Journal:
 def __init__(self,path):self.path=Path(path);self.stream=self.path.open('x');self.count=0;self.diagnostics=set()
 def append(self,event):self.stream.write(json.dumps(event)+'\n');self.stream.flush()
 def update(self,records,error):
  assert len(records)in [self.count,self.count+1]
  if len(records)>self.count:self.append(dict(type='query',record=records[-1]));self.count+=1
  if records:
   r=records[-1];key=(r['case'],r['index'])
   d={k:r[k]for k in ['K_stage_outside_benchmark','K_stage_diagnostic_error']if k in r}
   if d and key not in self.diagnostics:self.append(dict(type='diagnostic',case=key[0],index=key[1],fields=d));self.diagnostics.add(key)
  if error:self.append(dict(type='error',error=error,completed_queries=self.count))
 def close(self):self.stream.close()
def aggregate(path):
 rows=[];lookup={};errors=[]
 for line in Path(path).read_text().splitlines():
  e=json.loads(line)
  if e['type']=='query':r=e['record'];lookup[(r['case'],r['index'])]=r;rows.append(r)
  elif e['type']=='diagnostic':lookup[(e['case'],e['index'])].update(e['fields'])
  elif e['type']=='error':errors.append(e)
 return dict(records=rows,errors=errors)
