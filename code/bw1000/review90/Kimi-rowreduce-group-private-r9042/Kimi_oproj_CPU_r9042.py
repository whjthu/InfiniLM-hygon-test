import pathlib,json,hashlib,subprocess,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-rowreduce-group-r9042';b.mkdir(exist_ok=False);(b/'src').mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'changed_HOST_TUs':1,'unchanged_shim_relink':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 parent=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_text());cmd=parent['commands'][0][:];cmd.insert(1,'-I/opt/dtk/include');src=b/'src/delta.cpp';src.write_bytes((r/'incoming/Kimi_oproj_delta_r9042.cpp').read_bytes());helper=r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp';(src.parent/helper.name).write_bytes(helper.read_bytes());cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(b/'delta.o');cmd[cmd.index('-MF')+1]=str(b/'delta.d');trace=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());link=trace['commands'][-1][:];old='/refactor/candidates/Kimi-CCL512-r8606/delta.o';assert link.count(old)==1;link[link.index(old)]=str(b/'delta.o');link.extend(['-L/opt/dtk/lib','-lrccl']);main=b/'libKimi_oproj_Main_r9042.so';link[link.index('-o')+1]=str(main)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map,')or a.startswith('-Wl,-Map='):link[j]='-Wl,-Map,'+str(b/'link.map')
 shimold=json.loads((r/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json').read_text());shim=shimold['command'][:];shim[shim.index(shimold['actual678_DSO'])]=str(main);native=b/'libKimi_oproj_shim_r9042.so';shim[shim.index('-o')+1]=str(native);q['commands']=[cmd,link,shim]
 for j,c in enumerate(q['commands']):
  p=subprocess.run(c,capture_output=True,text=True,timeout=140);(b/f'command{j}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-6000:]
 reloc=subprocess.check_output(['readelf','-rW',str(native)],text=True);slots={}
 for name in shimold['provider_relocation_slots']:
  rows=[x.split()for x in reloc.splitlines()if name in x and('JUMP_SLOT'in x or'GLOB_DAT'in x)];assert len(rows)==1;slots[name]=int(rows[0][0],16)
 expected={k:(sha(main)if 'KimiK3DeltaAttention'in k else shimold['provider_DSO_SHA256'])for k in slots};needed=[x for x in subprocess.check_output(['readelf','-d',str(native)],text=True).splitlines()if'NEEDED'in x];assert str(main)in needed[0];mainreloc=subprocess.check_output(['readelf','-rW',str(main)],text=True);groupslots={}
 for name in ['ncclGroupStart','ncclGroupEnd']:
  rr=[x.split()for x in mainreloc.splitlines()if name in x and('JUMP_SLOT'in x or'GLOB_DAT'in x)];assert len(rr)==1;groupslots[name]=int(rr[0][0],16)
 rccl=pathlib.Path('/opt/dtk/lib/librccl.so.1');assert sha(rccl)=='32aa13da8bbd4d9a4ec37be08014f0783f5f576c7439dc10a3e1c661aa35427a';q['RCCL']={'path':str(rccl),'SHA256':sha(rccl),'Main_GOT_group_slots':groupslots};q.update(completed=True,native=str(native),native_SHA256=sha(native),candidate_Main=str(main),candidate_Main_SHA256=sha(main),actual678_DSO=str(main),actual678_DSO_SHA256=sha(main),provider_DSO=shimold['provider_DSO'],provider_DSO_SHA256=shimold['provider_DSO_SHA256'],provider_relocation_slots=slots,provider_expected_SHA256=expected,delta_source_SHA256=sha(src),delta_object_SHA256=sha(b/'delta.o'),math_header_SHA256=sha(helper),DT_NEEDED=needed)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-oproj-CPU-r9042-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
