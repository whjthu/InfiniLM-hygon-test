import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=r/'results/hardware-dumma-CPU-r8403-launch.json';assert not p.exists()
a=['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c4','-w','a02r4n04','--gres=none','--mem=4000M','-t','00:10:00',
 '--job-name=hardware-dumma-CPU-r8403','--output='+str(r/'logs/hardware-dumma-CPU-r8403-%J.log'),
 'env','WAVE_INSIDE_R2323=hardware_dumma_CPU_r8403_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with p.open('x')as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,args=a),f,indent=2)
with(r/'logs/hardware-dumma-CPU-r8403-controller.log').open('x')as f:z=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_only=True,pid=z.pid,record=str(p))))
