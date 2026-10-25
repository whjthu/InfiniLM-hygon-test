import pathlib,json,zlib,base64
r=pathlib.Path(__file__).resolve().parent;b=r/'Kimi-projection-real-input-private-r9030';old=(b/'CPU_build_fixed.py').read_text();code=old.replace("b=r/'candidates/Kimi-projection-real-input-r9030b'","b=r/'candidates/Kimi-projection-real-input-r9030c'").replace("'libKimi_projection_real_r9030b.so'","'libKimi_projection_real_r9030c.so'").replace("q['command']=cmd","common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so';assert sha(common)=='2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2';cmd.insert(cmd.index('-shared')+2,str(common));q['command']=cmd").replace("'Kimi-projection-CPU-r9030b-'","'Kimi-projection-CPU-r9030c-'")
# Actual provider, not wrapper Model DSO, owns constructor/vtable/forward.
code=code.replace("provider_DSO='/refactor/candidates/Kimi-ordinary-pair-r9007/libKimi_pair_r9007.so',provider_DSO_SHA256=sha(r/'candidates/Kimi-ordinary-pair-r9007/libKimi_pair_r9007.so')","provider_DSO=str(common),provider_DSO_SHA256=sha(common)")
compile(code,'build','exec');(b/'CPU_build_common.py').write_bytes(code.encode());inside=(b/'CPU_inside_fixed.sh').read_text().replace('Kimi_projection_CPU_fixed_r9030.py','Kimi_projection_CPU_common_r9030.py');files={'Kimi_projection_CPU_common_r9030.py':code,'Kimi_projection_CPU_common_r9030_inside.sh':inside};payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists(),n;p.write_bytes(s.encode())
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-projection-CPU-r9030c','--output='+str(r/'logs/Kimi-projection-CPU-r9030c-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_projection_CPU_common_r9030_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-projection-CPU-r9030c-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args}))
""".replace('PAYLOAD',repr(payload));(r/'invoke_Kimi_projection_CPU_common_r9030.py').write_bytes(remote.encode())
