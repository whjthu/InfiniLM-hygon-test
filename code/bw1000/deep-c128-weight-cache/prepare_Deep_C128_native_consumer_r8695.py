import pathlib
r=pathlib.Path(__file__).parent
s=(r/'deep_C128_component_r8688.py').read_text().replace('r8688','r8695').replace('deep-C128-component-r8686','deep-C128-component-r8694').replace('libdeep_C128_component_r8686.so','libdeep_C128_component_r8694.so')
s=s.replace("LIB_SHA='7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2'","LIB_SHA='PENDING_CPU_SHA'").replace('a93e8c173cc89ab81a9eec0568a9e3d8df53218d53279726a424b2e3483645ce','PENDING_CPU_RECEIPT')
s=s.replace("ARMS=['O','N']+['O','N','N','O','O','N','N','O']","ARMS=['O','N']+['O','N','N','O']*3").replace('list(range(2,10))','list(range(2,14))')
s=s.replace('lib.modern_real_compress_release_workers.argtypes=[]','lib.modern_real_compress_release_workers.argtypes=[]\nlib.modern_real_compress_timing_r8694.argtypes=[ctypes.POINTER(ctypes.c_double),S];lib.modern_real_compress_timing_r8694.restype=ctypes.c_int')
s=s.replace('self.closed=False','self.closed=False;self.native_slot=(ctypes.c_double*3)();self.native_sequence=0')
s=s.replace('return [score,pooled,rotated],wall,thread_cpu','check(lib.modern_real_compress_timing_r8694(self.native_slot,3))\n  native,sequence,native_tokens=list(self.native_slot);assert sequence==self.native_sequence+1 and native_tokens==n and 0<native<=wall\n  self.native_sequence=int(sequence)\n  return [score,pooled,rotated],wall,thread_cpu,native')
s=s.replace('out,wall,cpu=self.forward','out,wall,cpu,native=self.forward').replace("'owner_thread_CPU_s':cpu,'validation'","'owner_thread_CPU_s':cpu,'native_inner_wall_s':native,'outer_minus_inner_s':wall-native,'validation'")
(r/'deep_C128_component_r8695.py').write_text(s,newline='\n')
s=(r/'verify_Deep_C128_component_r8688.py').read_text().replace("arms=['O','N']+['O','N','N','O','O','N','N','O']","arms=['O','N']+['O','N','N','O']*3").replace('==20','==28').replace('range(10)','range(14)').replace('for c in [0,1]:','for c in [0,1,2]:').replace('7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2','PENDING_CPU_SHA')
s=s.replace("'slowest_owner_CAPI_wall_s':max(x['native_enclosing_wall_s']for x in r)","'slowest_owner_CAPI_wall_s':max(x['native_enclosing_wall_s']for x in r),'slowest_native_inner_wall_s':max(x['native_inner_wall_s']for x in r),'outer_minus_inner_owner_range_s':[min(x['outer_minus_inner_s']for x in r),max(x['outer_minus_inner_s']for x in r)]")
a="'OFF_endpoint_drift_fraction':abs(rows[3]['slowest_owner_CAPI_wall_s']/rows[0]['slowest_owner_CAPI_wall_s']-1)"
b=a+",'native_OFF_s':statistics.mean(x['slowest_native_inner_wall_s']for x in rows if x['arm']=='O'),'native_ON_s':statistics.mean(x['slowest_native_inner_wall_s']for x in rows if x['arm']=='N'),'native_paired_speedup':statistics.mean(x['slowest_native_inner_wall_s']for x in rows if x['arm']=='O')/statistics.mean(x['slowest_native_inner_wall_s']for x in rows if x['arm']=='N'),'native_OFF_endpoint_drift_fraction':abs(rows[3]['slowest_native_inner_wall_s']/rows[0]['slowest_native_inner_wall_s']-1)"
assert a in s;s=s.replace(a,b);(r/'verify_Deep_C128_component_r8695.py').write_text(s,newline='\n')
for old,new in [('Deep_C128_component_r8688_inside.sh','Deep_C128_component_r8695_inside.sh'),('launch_Deep_C128_component_r8688.py','launch_Deep_C128_component_r8695.py')]:
 (r/new).write_text((r/old).read_text().replace('8688','8695'),newline='\n')
print('3ABBA consumer/entry draft ready; waiting exact CPU SHA, no GPU')
