import pathlib,json,subprocess,hashlib,os,time,traceback,shlex
r=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;b=r/'candidates/Kimi-current-MoE-profile-r9264';b.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'new_HOST_TUs':1,'new_GPU_math_compile':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
try:
 current=json.loads((r/'candidates/Kimi-shard-only-Model-r9183/CPU-build.json').read_bytes());assert current['completed']and sha(current['native'])==current['native_SHA256']=='b4f916dcf32f3bafb7451c94d6a98746a0e6c084edf73aeb6e229275579dbb8d'
 for p,h in current['all_linked_object_SHA256'].items():assert sha(p)==h,p
 source=json.loads((r/'candidates/Kimi-I192-batch-Model-r9129/CPU-build.json').read_bytes());assert source['completed']and source['moe_object_SHA256']=='55fb31b5f4fa1510f06e8b62894d64d52926862371f43e1e2d8c5aaeb2624800'
 old=json.loads((r/'candidates/Kimi-MoE-event-profile-r9220e/CPU-build.json').read_bytes())
 cmd=old['commands'][-1][:]
 cmd=[str(D/'current_moe_runner_r9264.cpp')if x.endswith('/kimi_TP16_event_runner_r9220.cpp')else current['native']if x==old['native']else str(b/'adapter.d')if x.endswith('/adapter.d')else str(b/'libKimi_current_moe_runner_r9264.so')if x==old['library']else x for x in cmd]
 cmd.insert(1,'-DINFINICORE_RANDOM_SAMPLE_OMITTED')
 q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=220);(b/'compile.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-9000:]
 library=b/'libKimi_current_moe_runner_r9264.so'
 env=dict(os.environ,LD_LIBRARY_PATH=old['LD_LIBRARY_PATH']);z=subprocess.run(['ldd','-r',str(library)],env=env,text=True,capture_output=True,timeout=60);assert z.returncode==0 and'not found'not in z.stdout+z.stderr and'undefined symbol:'not in z.stdout+z.stderr,z.stdout+z.stderr
 md=(b/'adapter.d').read_text().replace('\\\n',' ');paths=shlex.split(md.split(':',1)[1]);project=[p for p in paths if p.startswith('/refactor/')and pathlib.Path(p).is_file()]
 actual_headers={p:sha(p)for p in project};critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert critical in actual_headers
 sealed=json.loads((r/'results/Kimi-Model-events-CPU-r8669zq1.json').read_bytes());historical_headers={}
 for v in sealed['fresh_MD'].values():historical_headers.update(v)
 assert historical_headers[critical]==actual_headers[critical],'current ctor-header ABI cannot be silently assumed'
 deps=subprocess.check_output(['readelf','-d',str(library)],text=True);assert current['native']in deps
 defined=subprocess.check_output(['nm','-D','-C','--defined-only',str(library)],text=True);assert'modern_kimi_current_moe_run_r9264'in defined
 assert'KimiK3MoE::forward('not in defined,'no hidden new MoE math implementation'
 undefined=subprocess.check_output(['nm','-D', '--undefined-only',str(library)],text=True);cpp_moe=[x.split()[-1]for x in undefined.splitlines()if'KimiK3MoE'in x];assert cpp_moe
 defined_native=subprocess.check_output(['nm','-D','--defined-only',current['native']],text=True)
 assert all(x in defined_native for x in cpp_moe),cpp_moe
 # CPU dtype/pointer geometry contract uses mock addresses only; no data pointer is dereferenced.
 expected=json.loads((D/'expected-headers-r9125.json').read_bytes());assert expected
 q.update(completed=True,native=current['native'],native_SHA256=current['native_SHA256'],library=str(library),library_SHA256=sha(library),current_objects=current['all_linked_object_SHA256'],LD_LIBRARY_PATH=old['LD_LIBRARY_PATH'],ldd_relocation=z.stdout+z.stderr,wrapper_DT_NEEDED=deps,undefined_MoE_symbols_bound_to_b4=cpp_moe,actual_headers=actual_headers,critical_MoE_header_SHA256=actual_headers[critical],ABI_header_matches_actual_decoder_ctor_sealed_MD=True,source_SHA256={p.name:sha(p)for p in D.iterdir()if p.is_file()and p.suffix in ['.cpp','.py','.json','.sh']},actual_profile_trace_not_qualified=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(b/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','native_SHA256','library_SHA256','error']}));assert q['completed']
