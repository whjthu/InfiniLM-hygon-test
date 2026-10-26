import collections

def union_us(intervals):
 merged=[]
 for a,b in sorted(intervals):
  if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 return sum(b-a for a,b in merged)

def summarize(trace):
 events=trace.get('traceEvents',[]);whole=[e for e in events if e.get('name')=='GLM_CHAIN_DIAGNOSTIC'and e.get('ph')=='X'];assert len(whole)==1,whole
 lo=whole[0]['ts'];hi=lo+whole[0]['dur'];kernels=[e for e in events if e.get('ph')=='X'and 'kernel'in str(e.get('cat','')).lower()and lo<=e.get('ts',-1)<=hi];copies=[e for e in events if e.get('ph')=='X'and any(s in str(e.get('cat','')).lower()for s in ['gpu_memcpy','gpu_memset'])and lo<=e.get('ts',-1)<=hi];byname=collections.defaultdict(lambda:{'calls':0,'sum_seconds':0.})
 for e in kernels:
  z=byname[e['name']];z['calls']+=1;z['sum_seconds']+=e['dur']/1e6
 device_intervals=[(e['ts'],e['ts']+e['dur'])for e in kernels+copies];kernel_intervals=[(e['ts'],e['ts']+e['dur'])for e in kernels]
 return {'kernel_count':len(kernels),'copy_memset_count':len(copies),'kernel_interval_union_seconds':union_us(kernel_intervals)/1e6,'kernel_plus_copy_interval_union_seconds':union_us(device_intervals)/1e6,'kernel_sum_seconds':sum(e['dur']for e in kernels)/1e6,'trace_host_scope_seconds':whole[0]['dur']/1e6,'kernel_by_name':dict(sorted(byname.items(),key=lambda x:-x[1]['sum_seconds'])),'kernel_device_ids':sorted(set(str(e.get('args',{}).get('device','UNKNOWN'))for e in kernels)),'definition':'activity interval union within one CPUquery scope; not hardware occupancy; host enqueue spans not GPUbusy; no Model/CCL'}
