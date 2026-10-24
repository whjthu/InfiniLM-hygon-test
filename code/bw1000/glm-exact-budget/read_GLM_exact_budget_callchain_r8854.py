import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q=json.loads((r/'candidates/glm-exact-budget-r8854/CPU-build.json').read_text());lib=pathlib.Path(q['library'].replace('/refactor/',str(r)+'/'));symtext=subprocess.check_output(['nm','-S','--defined-only',str(lib)],text=True,timeout=30);picked=[x.split()for x in symtext.splitlines()if 'glm_resident_exact_r8854'in x or(x.split()and x.split()[-1]=='modern_glm_model_resident_mask_r5640')];out=[]
for sym in picked:
 if len(sym)<4:continue
 start=int(sym[0],16);size=int(sym[1],16);s=subprocess.check_output(['objdump','-d','--start-address='+str(start),'--stop-address='+str(start+size),str(lib)],text=True,timeout=30)
 lines=s.splitlines();i=next((i for i,x in enumerate(lines)if 'cmp    $0x20,%rax'in x),0);out.append({'rows32_branch_and_budget':lines[max(0,i-4):i+28],'symbol':sym[-1],'bytes':size,'SHA256':hashlib.sha256(s.encode()).hexdigest(),'instructions':s if 'glm_resident_exact_r8854'in sym[-1]else'\n'.join(x for x in s.splitlines()if any(v in x for v in('call','shl','imul','0x20,','lea ')))})
print(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'DSO_SHA256':q['library_SHA256'],'actual_linked_callchain':out}))
