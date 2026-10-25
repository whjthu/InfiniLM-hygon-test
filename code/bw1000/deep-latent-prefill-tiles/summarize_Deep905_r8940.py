import pathlib,json,statistics,math
r=pathlib.Path.cwd();d=json.loads((r/'Deep-latent-prefill-component-watch-r8939-b.json').read_bytes());q=d['report'];arms=[dict(arm=t['arm'],max_owner_ms=max(v['ms']for v in t['owners']),owners=t['owners'])for t in q['timed']];x={a:[v['max_owner_ms']for v in arms if v['arm']==a]for a in [0,1]};quality=[]
for c in q['quality']:
 quality.append(dict(label=c['label'],nrmse_range=[min(v['nrmse']for v in c['error']),max(v['nrmse']for v in c['error'])],maxabs=max(v['maxabs']for v in c['error']),p99abs=max(v['p99abs']for v in c['error']),bit_equal_all=all(v['bit_equal']for v in c['error']),counter_deltas=[v['counter_delta']for v in c['candidate']]))
blocks=[]
for i in [0,4]:
 z=arms[i:i+4];a=statistics.mean(v['max_owner_ms']for v in z if v['arm']==0);c=statistics.mean(v['max_owner_ms']for v in z if v['arm']==1);blocks.append(dict(A_ms=a,C_ms=c,A_over_C=a/c))
out=dict(maxowner_ms_by_arm=x,means={str(a):statistics.mean(v)for a,v in x.items()},A_over_C=statistics.mean(x[0])/statistics.mean(x[1]),balanced_blocks=blocks,quality=quality,ownrepeat=q['ownrepeat']);p=r/'Deep905-numeric-r8940.json';p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))