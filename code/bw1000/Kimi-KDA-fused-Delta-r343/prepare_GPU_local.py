from pathlib import Path
import ast
D=Path(__file__).resolve().parent;S=D.parent/'Kimi-C1-Delta-GPU-r310'
code=(S/'driver.py').read_text().replace("S=R/'incoming/Kimi-C1-Delta-profile-r309'","S=D")
code=code.replace("D/'SOURCE_MANIFEST.json'","D/'GPU_MANIFEST.json'").replace('KIMI_R310_MANIFEST_SHA','KIMI_R343_GPU_MANIFEST_SHA')
code=code.replace("candidates/Kimi-C1-Delta-profile-r309/CPU-build.json","candidates/Kimi-KDA-fused-Delta-r343/CPU-build.json").replace('Kimi-C1-Delta-GPU-r310','Kimi343-fused-Delta').replace('KIMI_C1_DELTA_R310_NORMAL','KIMI343_NORMAL')
code=code.replace("    lib.layer0_profile_mode_r9140.argtypes=[ctypes.c_void_p,ctypes.c_int]","    lib.layer0_profile_mode_r9140.argtypes=[ctypes.c_void_p,ctypes.c_int]\n    lib.layer0_fused_mode_r343.argtypes=[ctypes.c_void_p,ctypes.c_int]\n    lib.layer0_fused_counts_r343.argtypes=[ctypes.c_void_p,ctypes.c_void_p]")
code=code.replace("            for arm,on in [('OFF-before',False),('ON',True),('OFF-after',False)]:", "            schedule=[(r,on)for r in range(8)for on in ([False,True,True,False]if r%2==0 else[True,False,False,True])]\n            for block,(round_id,on) in enumerate(schedule):\n                arm='ON'if on else'OFF'")
code=code.replace('check(lib.layer0_profile_mode_r9140(ctx,int(on)))','check(lib.layer0_profile_mode_r9140(ctx,0));check(lib.layer0_fused_mode_r343(ctx,int(on)))')
code=code.replace("profile=json.loads(lib.layer0_profile_last_r9140(ctx))if on else None","profile=None")
old="                    assert equal,'Observer changed output or recurrent state'\n                    if on:assert profile['C']==profile['T']==1 and profile['pool']==3 and profile['added_segment_syncs']==0"
new="""                    def f32(v):return (v.astype(np.uint32)<<16).view(np.float32)
                    diag={}
                    for name,old_value,new_value in [('output',reference[0],output_value),('conv',reference[1][0],updated[0]),('state',reference[1][1],updated[1])]:
                        aa,bb=f32(old_value),f32(new_value);assert np.isfinite(aa).all()and np.isfinite(bb).all()
                        diag[name]={'max_abs':float(np.max(np.abs(aa-bb))),'rms':float(np.sqrt(np.mean((aa-bb)**2)))}
                        assert np.allclose(aa,bb,atol=.05,rtol=.02),(name,diag[name])
                    if not on:assert equal,'baseline self drift in output/state'
"""
assert old in code;code=code.replace(old,new)
code=code.replace("{'arm':arm,'trial':trial", "{'arm':arm,'block':block,'round':round_id,'numerical':diag,'trial':trial")
code=code.replace('layer_calls=31','layer_calls=321').replace("len(w['samples'])==30","len(w['samples'])==320")
code=code.replace("            own.update(completed=True", "            counts=(ctypes.c_uint64*2)();check(lib.layer0_fused_counts_r343(ctx,counts));own['fused_counts']=list(counts);assert counts[0]==160 and counts[1]==0,('fused eligibility failed',list(counts))\n            own.update(completed=True")
ast.parse(code);(D/'driver.py').write_bytes(code.encode())
inside=(S/'GPU_inside.sh').read_text().replace('Kimi-C1-Delta-GPU-r310','Kimi-KDA-fused-Delta-r343')
(D/'GPU_inside.sh').write_bytes(inside.encode())
print('GPU driver ready, no launch')
