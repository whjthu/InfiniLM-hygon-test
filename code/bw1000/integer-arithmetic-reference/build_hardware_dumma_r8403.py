"""CPU-only compile and target ISA inspection; never dlopen/device init."""
import pathlib,hashlib,json,shutil,subprocess,time,datetime,os,re
r=pathlib.Path('/refactor');base=r/'candidates/hardware-dumma-r8403'
assert not base.exists();base.mkdir();start=time.perf_counter()
src=r/'incoming/hardware_dumma_r8403.hip';compiler=shutil.which('hipcc');assert compiler
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report=dict(completed=False,CPU_only=True,GPU_initialized=False,producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
 source_SHA256=sha(src),compiler=compiler,commands=[])
def run(args,name):
 report['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=300)
 (base/(name+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(name,p.returncode,p.stderr[-4500:])
try:
 lib=base/'libhardware_dumma_r8403.so';asm=base/'hardware_dumma_r8403.s'
 run([compiler,'-std=c++17','-O3','--offload-arch=gfx936','-fPIC','-shared',str(src),'-o',str(lib)],'shared')
 run([compiler,'-std=c++17','-O3','--offload-arch=gfx936','--cuda-device-only','-S',str(src),'-o',str(asm)],'assembly')
 lines=asm.read_text().splitlines();mmac=[s.strip()for s in lines if re.search(r'\b(?:v_|s_)?mmac',s)]
 assert mmac,'No target MMAC instruction in compiled device assembly'
 report.update(completed=True,library=str(lib),library_SHA256=sha(lib),assembly=str(asm),assembly_SHA256=sha(asm),
  mmac_instruction_lines=mmac,elapsed_seconds=time.perf_counter()-start,
  header_SHA256={n:sha(pathlib.Path('/opt/dtk/include')/n)for n in ['du_mma.h','du_mma.hpp']},
  scope='Compiler+ISA only; no runtime numerical or throughput validation, no physical peak/model speedup.')
except BaseException as exc:
 report['error']=repr(exc);raise
finally:
 report['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (base/'CPU-build.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items()if k not in ['commands','mmac_instruction_lines']}))
