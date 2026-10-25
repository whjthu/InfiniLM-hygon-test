import pathlib,json,subprocess,hashlib,shlex,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-delta-selector-r9027';b.mkdir(exist_ok=False);(b/'src').mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'native_host_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 parent=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_text());cmd=parent['commands'][0][:];source=b/'src/kimi_k3_delta_attention_serial_r9027.cpp';source.write_bytes((r/'incoming/kimi_k3_delta_attention_serial_r9027.cpp').read_bytes())
 helper=r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp';(b/'src'/helper.name).write_bytes(helper.read_bytes());q['actual_math_header_SHA256']=sha(helper)
 oldsrc=cmd[cmd.index('-c')+1];assert sha(oldsrc)=='787af2aac64ae58dedfa2654f8efd94cc449c1aab7de03549b16789fed3898cd';cmd[cmd.index('-c')+1]=str(source);cmd[cmd.index('-o')+1]=str(b/'delta.o');cmd[cmd.index('-MF')+1]=str(b/'delta.d');q['command']=cmd
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=150);(b/'compile.log').write_text(p.stdout+p.stderr);q['compile_returncode']=p.returncode;assert p.returncode==0,p.stderr[-7000:]
 deps=shlex.split((b/'delta.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);nm=subprocess.check_output(['nm','-C','--defined-only',str(b/'delta.o')],text=True);assert 'modern_kimi_KDA_serial_counts_r9027'in nm and 'KimiK3DeltaAttention::forward('in nm
 q.update(completed=True,source_SHA256=sha(source),object_SHA256=sha(b/'delta.o'),dependencies_SHA256={n:sha(n)for n in deps},scope='compile-only one actualModel delta HOSTTU; no link/dlopen/GPU/Model, oldflags and math helper unchanged')
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
p=r/'results'/('Kimi-delta-selector-CPU-r9027-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['dependencies_SHA256']}));assert q['completed']
