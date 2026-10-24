"""CPU-only compile and target ISA inspection; never dlopen/device init."""
import pathlib,hashlib,json,shutil,subprocess,time,datetime,os,re
r=pathlib.Path('/refactor');base=r/'candidates/hardware-fp32-fma-r8800'
assert not base.exists();base.mkdir();start=time.perf_counter()
src=r/'incoming/hardware_fp32_fma_r8800.hip';compiler=shutil.which('hipcc');assert compiler
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report=dict(completed=False,CPU_only=True,GPU_initialized=False,producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
 source_SHA256=sha(src),compiler=compiler,commands=[])
def run(args,name):
 report['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=300)
 (base/(name+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(name,p.returncode,p.stderr[-4500:])
try:
 lib=base/'libhardware_fp32_fma_r8800.so';asm=base/'hardware_fp32_fma_r8800.s'
 run([compiler,'-std=c++17','-O3','--offload-arch=gfx936','-fPIC','-shared',str(src),'-o',str(lib)],'shared')
 run([compiler,'-std=c++17','-O3','--offload-arch=gfx936','--cuda-device-only','-S',str(src),'-o',str(asm)],'assembly')
 lines=asm.read_text().splitlines();fma=[s.strip()for s in lines if re.search(r'\bv_fma_f32\b',s)];matrix=[s.strip()for s in lines if re.search(r'\b(?:v_)?(?:mmac|mfma|wmma)',s)]
 assert len(fma)==8 and not matrix,('Expectedexact8v_fma_f32staticloopinstructionsandnomatrix',fma,matrix)
 report.update(completed=True,library=str(lib),library_SHA256=sha(lib),assembly=str(asm),assembly_SHA256=sha(asm),fma_instruction_lines=fma,matrix_instruction_lines=matrix,elapsed_seconds=time.perf_counter()-start,scope='Compiler+ISAonly,noGPU/dlopen/numerical/throughput.8independentFP32FMAregisterchains,notphysicalpeakorModelTPS.')

except BaseException as exc:
 report['error']=repr(exc);raise
finally:
 report['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (base/'CPU-build.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items()if k not in ['commands','mmac_instruction_lines']}))
