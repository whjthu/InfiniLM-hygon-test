"""r347 passive KFD/ports plus actual bootstrap PID and TIME_WAIT diagnostics."""
from pathlib import Path
import argparse,datetime,json,os,socket,subprocess,time
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','post','failure-post'],required=True);a=p.parse_args()
 node=socket.gethostname().split('.')[0];assert node in ['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
 def snapshot():
  kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();pids=sorted(int(x.name)for x in kfd.iterdir()if x.name.isdigit())
  devices=[]
  for k in sorted(Path('/sys/class/kfd/kfd/topology/nodes').iterdir()):
   props={v[0]:v[1]for l in(k/'properties').read_text().splitlines()if len(v:=l.split())==2}
   if int((k/'gpu_id').read_text())==0:continue
   render=int(props['drm_render_minor']);pci=(Path('/sys/class/drm')/('renderD'+str(render))/'device').resolve().name
   assert ':'in pci and'.'in pci;devices.append(dict(KFD_node=int(k.name),gpu_id=int((k/'gpu_id').read_text()),drm_render_minor=render,PCI=pci,location_id=props.get('location_id')))
  assert len(devices)==8 and len({v['PCI']for v in devices})==8
  matching=[]
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
   try:args=(proc/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if b'/refactor/incoming/Kimi-TP8PP4-ordinary-r410/runtime-v1/bootstrap.py'in args:matching.append(int(proc.name))
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
  q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=pids,KFD_empty=not pids,devices=devices,unique_PCI_count=8,matching_bootstrap_PIDs=matching,ports_bindable=bindable,port_entries=entries,no_LISTEN=no_listen,post_only_TIME_WAIT_or_empty=post_socket_closed,ib_ipv4=ips,passed=passed,phase=a.phase,TCP_connections_made=0)
  return q
 started=time.monotonic()
 while True:
  q=snapshot();q['passive_post_wait_s']=time.monotonic()-started
  if a.phase=='before'or q['passed']or time.monotonic()-started>=90:break
  time.sleep(.5)
 print(json.dumps(q));assert q['passed']
if __name__=='__main__':main()
