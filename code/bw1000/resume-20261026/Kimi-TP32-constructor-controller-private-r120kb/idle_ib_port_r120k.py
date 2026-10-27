"""Owned-node CPU probe: no CUDA/Runtime import or process attachment."""
import pathlib,os,json,socket,datetime,fcntl,struct,sys
port=int(sys.argv[1]);check_port=sys.argv[2]=='before';assert 1024<port<65535
p=pathlib.Path('/sys/class/kfd/kfd/proc');assert p.exists();pids=sorted(x.name for x in p.iterdir()if x.name.isdigit());assert not pids,pids
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as sock:reply=fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',b'ib0'))
assert struct.unpack_from('H',reply,16)[0]==socket.AF_INET
address=socket.inet_ntoa(reply[20:24]);assert address.startswith('12.')
if check_port:
 with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as sock:sock.bind(('0.0.0.0',port))
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'kfd_pids':pids,'ib0_ipv4':[address],'rendezvous_port':port,'port_checked':check_port,'wildcard_port_bind_free':True if check_port else None,'passed':True,'CUDA_initialized':False,'IPv4_source':'Linux SIOCGIFADDR ib0'}))
