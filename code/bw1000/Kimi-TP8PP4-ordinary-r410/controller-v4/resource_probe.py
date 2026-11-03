"""r347 passive KFD/ports plus actual bootstrap PID and TIME_WAIT diagnostics."""
from pathlib import Path
import argparse,datetime,json,os,socket,subprocess,time,re
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','post','failure-post'],required=True);a=p.parse_args()
 node=socket.gethostname().split('.')[0];assert node in ['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
 def snapshot():
  kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();pids=sorted(int(x.name)for x in kfd.iterdir()if x.name.isdigit())
  devices=[];optional_KFD_fields=[]
  for render in sorted(Path('/sys/class/drm').glob('renderD*/device')):
   pci=render.resolve().name;assert re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}[.][0-7]',pci)
   fields={}
   for key in ['vendor','device','class']:
    try:fields[key]=dict(value=(render/key).read_text().strip())
    except OSError as e:fields[key]=dict(error=repr(e))
   driver=(render/'driver').resolve().name
   assert fields['vendor'].get('value')=='0x1d94'and fields['device'].get('value')=='0x6320'and fields['class'].get('value')=='0x0b4000'and driver=='hycu',fields
   devices.append(dict(PCI=pci,render=render.parent.name,driver=driver,fields=fields,inventory_source='DRM_renderD_device_and_readable_PCI_ids'))
  assert len(devices)==8 and len({v['PCI']for v in devices})==8
  for k in sorted(Path('/sys/class/kfd/kfd/topology/nodes').iterdir()):
   row=dict(KFD_node=k.name)
   for key in ['gpu_id','properties']:
    try:row[key]=dict(value=(k/key).read_text().strip())
    except OSError as e:row[key]=dict(error=repr(e))
   optional_KFD_fields.append(row)
  matching=[]
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
   try:args=(proc/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if b'/refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v3/bootstrap.py'in args:matching.append(int(proc.name))
  ports=[32947,39947];bindable={};entries=[]
  for port in ports:
   with socket.socket()as s:
    try:s.bind(('0.0.0.0',port));bindable[str(port)]=True
    except OSError:bindable[str(port)]=False
  for table in ['/proc/net/tcp','/proc/net/tcp6']:
   for line in Path(table).read_text().splitlines()[1:]:
    v=line.split();local=int(v[1].split(':')[-1],16);remote=int(v[2].split(':')[-1],16)
    if local in ports or remote in ports:entries.append(dict(table=table,local_port=local,remote_port=remote,state=v[3],inode=v[9]))
  raw=subprocess.check_output(['ip','-j','-4','address'],text=True,timeout=5);ips=sorted({v['local']for r in json.loads(raw)if r['ifname'].startswith('ib')for v in r.get('addr_info',[])if v.get('scope')=='global'})
  no_listen=not any(v['state']=='0A'and v['local_port']in ports for v in entries)
  post_socket_closed=all(v['state']=='06'for v in entries)
  passed=not pids and not matching and bool(ips)and(all(bindable.values())if a.phase=='before'else no_listen and post_socket_closed and all(bindable.values()))
  q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=pids,KFD_empty=not pids,devices=devices,optional_KFD_fields=optional_KFD_fields,unique_PCI_count=8,matching_bootstrap_PIDs=matching,ports_bindable=bindable,port_entries=entries,no_LISTEN=no_listen,post_only_TIME_WAIT_or_empty=post_socket_closed,ib_ipv4=ips,passed=passed,phase=a.phase,TCP_connections_made=0)
  return q
 started=time.monotonic()
 while True:
  q=snapshot();q['passive_post_wait_s']=time.monotonic()-started
  if a.phase=='before'or q['passed']or time.monotonic()-started>=90:break
  time.sleep(.5)
 print(json.dumps(q));assert q['passed']
if __name__=='__main__':main()
