import pathlib,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014/results/Kimi957-ISA-r8995';files=list(r.glob('*.isa.txt'));assert len(files)==1
s=files[0].read_text();s=s[s.index('<_ZN12_GLOBAL__N_16w13_a8'):];a=s.splitlines();rows=[]
for i,line in enumerate(a):
 if any(k in line for k in ['s_getpc_b64','global_load_dword','global_load_ubyte']):rows.append({'line':i,'context':a[max(0,i-3):i+4]})
print(json.dumps({'CPU_only':True,'rows':rows[:20]}))
