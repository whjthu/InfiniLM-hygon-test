import collections,datetime,hashlib,json,pathlib,re,statistics,subprocess
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/'o01-rccl-cross4x4-socket-shca-pair-v2-4e5075577e1446b1b5fd7064f2aaf273'
req='4e5075577e1446b1b5fd7064f2aaf273'; parent='1103606'
acct=subprocess.check_output(['sacct','-nP','-j',parent,'--format=JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'],text=True,timeout=8)
rows=[r.split('|') for r in acct.splitlines() if req in r]
answer=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),slurm=rows)
pin={}
def read(path):
 raw=path.read_bytes()
 pin[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
 return json.loads(raw)
arms={}
for index in range(4):
 d=root/f'arm-{index}'
 if not d.is_dir():continue
 mode=('socket','shca','shca','socket')[index]
 markers=collections.Counter(); channels=collections.Counter(); sample={}; errors=collections.Counter(); measures=0
 for node in ('f13r1n11','f13r1n12'):
  for i in range(8):
   child=d/node
   measures+=int((child/f'active-{i}.json').is_file())
   path=child/f'card-{i}.json'
   if path.is_file():
    a=read(path)
    if not a['error']:
     value,=a['results']
     assert value['payload_bytes']==26214400 and value['verified_elements']==13107200
     assert (a['node'],a['card'],a['mode'],a['arm'])==(node,i,mode,index)
     assert value['logical_rank']==(0 if node=='f13r1n11' else 4)+i%4
     assert value['network_policy']==('IBext_v8' if mode=='shca' else 'Socket')
     assert len(value['rank_max_seconds'])==96
     sample[node,i]=value
    else:errors[a['error']['kind']]+=1
   log=child/f'card-{i}.log'
   if log.is_file() and log.stat().st_size<512*1024:
    raw=log.read_bytes()
    markers.update({x.decode('ascii') for x in re.findall(rb'Using network (Socket|IBext_v8)\b',raw)})
    channels.update({x.decode('ascii') for x in re.findall(rb'via NET/(Socket|IBext(?:_v8)?)(?:/|\b)',raw)})
 result=dict(mode=mode,active_ranks=measures,passing_ranks=len(sample),errors=dict(errors),transport_markers=dict(markers),data_channel_markers=dict(channels))
 if len(sample)==16:
  series=[]
  for group in range(2):
   one=[v['rank_max_seconds'] for (_,i),v in sample.items() if i//4==group]
   assert len(one)==8 and all(x==one[0] for x in one)
   series.extend(one[0])
  result.update(paired_group_median_ms=1000*statistics.median(series),group_measurements=len(series))
 arms[str(index)]=result
answer['arms']=arms
f=root/'dispatch-terminal.json'
if f.is_file():
 d=read(f)
 answer.update(controller_finished=True,controller_failure=d['failure'],independent_post=bool(d['after'] and len(d['after']['nodes'])==2 and all(x['holders']==0 for x in d['after']['nodes'])),controller_accounted_successes=len(d['arms']))
else:
 answer.update(controller_finished=False)
log=root/'controller.log'
if log.is_file():
 b=log.read_bytes();answer['controller_log_SHA256']=hashlib.sha256(b).hexdigest()
 codes=re.findall(r'^(?:[A-Za-z]+Error): ([A-Z_0-9]+)',b.decode(errors='replace'),re.M)
 if codes:answer['last_controller_code']=codes[-1]
answer['receipt_pins_SHA256']=hashlib.sha256(json.dumps(pin,sort_keys=True).encode()).hexdigest()
print(json.dumps(answer,allow_nan=False))
