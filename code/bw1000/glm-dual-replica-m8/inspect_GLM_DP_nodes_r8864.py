import subprocess,json,datetime
allocation=subprocess.check_output(['scontrol','show','job','910371','-o'],text=True,timeout=25)
nodes=subprocess.check_output(['scontrol','show','hostnames',next(x.split('=',1)[1]for x in allocation.split()if x.startswith('NodeList='))],text=True,timeout=25).splitlines()
candidate=['a02r3n03','a02r3n07','a02r3n15','a02r3n16']
active=[]
for line in subprocess.check_output(['sacct','-S','2026-10-24','-j','910371','-n','-P','-o','JobID,JobName%100,State,NodeList%100'],text=True,timeout=25).splitlines():
 v=line.split('|')
 if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[2].split()[0]in('RUNNING','PENDING','CONFIGURING','COMPLETING'):active.append(v)
print(json.dumps({'CPU_read_only':True,'GPU':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_allocation':allocation,'allocated_nodes':nodes,'proposed_nodes':candidate,'proposed_in_allocation':set(candidate)<=set(nodes),'active_steps':active,'KFD':'NOT_QUERIED_NO_GPU_GO','new_Model':'NOT_SUBMITTED'}))
