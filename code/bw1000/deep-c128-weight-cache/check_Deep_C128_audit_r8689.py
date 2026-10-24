import pathlib,json,subprocess,sys
r=pathlib.Path(__file__).parent
out={'CPU_only':True,'GPU':False,'actual_verifier':True,'previous_Windows_negative_fixture_stderr_decode_warning_preserved':True}
for name,expected in [('Deep-C128-audit-CPU-fixture-r8688.json',0),('Deep-C128-audit-CPU-missing-row-r8688.json',1)]:
 z=subprocess.run([sys.executable,'-X','utf8',str(r/'verify_Deep_C128_component_r8688.py'),str(r/name)],capture_output=True,text=True,encoding='utf-8');assert z.returncode==expected,(z.returncode,z.stderr)
 if expected==0:
  d=json.loads(z.stdout);assert all(len(g['trials'])==8 and all(abs(c['paired_speedup']-1.25)<1e-9 for c in g['cycles'])for g in d['paired'].values())
 out[name]={'returncode':z.returncode,'expected':expected,'passed':True}
p=r/'Deep-C128-audit-CPU-regression-r8689.json'
with p.open('x')as f:json.dump(out,f,indent=2)
print(json.dumps(out))
