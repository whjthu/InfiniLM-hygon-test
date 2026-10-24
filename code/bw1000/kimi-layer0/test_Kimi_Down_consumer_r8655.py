from pathlib import Path
import importlib.util,json,copy,hashlib,numpy as np
r=Path('.');p=r/'kimi_down_runner_r8654.py';s=importlib.util.spec_from_file_location('down_cpu8655',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);manifest=json.loads((r/'Kimi-Down-operands-r8654.json').read_text());proof=m.validate_manifest(manifest);bad=copy.deepcopy(manifest);bad['ranks'][7]['cases'][1]['input']['shape'][1]=512
try:m.validate_manifest(bad)
except AssertionError:proof['missing_real_companions_rejected']=True
else:raise AssertionError('truncatedcompaniongeometryaccepted')
a=np.array([[0x3f80,0x4000]],dtype='<u2');b=a.copy();assert m.compare(a,b)['changed']==0;b[0,1]=0x4040;z=m.compare(a,b);assert z['changed']==1 and z['max_abs']==1.0 and z['finite'];proof.update(compare_fixture_only=True,GPU=False,passed=True,actual_consumer_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),note='CPU metadata/purecomparecontract; no GPU/numericalfaithfulness/performance.')
(r/'Kimi-Down-consumer-CPU-r8655.json').write_text(json.dumps(proof,indent=2),encoding='utf8',newline='\n');print(json.dumps(proof))
