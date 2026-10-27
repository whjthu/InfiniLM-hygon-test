import pathlib,os,json,socket,datetime,fcntl,struct
p=pathlib.Path('/sys/class/kfd/kfd/proc');assert p.exists();pids=sorted(x.name for x in p.iterdir()if x.name.isdigit());assert not pids,pids
# Linux SIOCGIFADDR: read address of the actual ib0 interface, no external ip binary.
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
 reply=fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',b'ib0'))
assert len(reply)>=24 and struct.unpack_from('H',reply,16)[0]==socket.AF_INET
address=socket.inet_ntoa(reply[20:24]);assert address.startswith('12.'),address
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'kfd_pids':pids,'ib0_ipv4':[address],'passed':True,'CUDA_initialized':False,'IPv4_source':'Linux SIOCGIFADDR ib0; no external binary'};
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as check:
 check.bind(('127.0.0.1',43844))
out['service_port_free']=True
print(json.dumps(out))
