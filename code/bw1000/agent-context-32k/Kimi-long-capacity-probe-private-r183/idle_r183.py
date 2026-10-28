"""CPU-only fresh owned-node KFD/IB/port check, no device library import."""
import pathlib,sys,socket,fcntl,struct,json,os,datetime
base=int(sys.argv[1]);tag=sys.argv[2];assert 1024<base<50000
p=pathlib.Path('/sys/class/kfd/kfd/proc');assert p.exists();pids=sorted(x.name for x in p.iterdir()if x.name.isdigit())
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as s:reply=fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'ib0'))
assert struct.unpack_from('H',reply,16)[0]==socket.AF_INET;address=socket.inet_ntoa(reply[20:24]);assert address.startswith('12.')
ports=[]
for port in [base,base+7000]:
 try:
  with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as s:s.bind(('0.0.0.0',port))
  ports.append({'port':port,'free':True})
 except OSError:ports.append({'port':port,'free':False})
q={'node':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tag':tag,'kfd_pids':pids,'ib0_ipv4':address,'ports':ports,'passed':not pids and all(x['free']for x in ports),'GPU_initialized':False};print(json.dumps(q));raise SystemExit(0 if q['passed']else 1)
