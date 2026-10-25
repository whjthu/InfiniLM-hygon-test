from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cpu=json.loads((r/'GLM-fragment-CPU-observed-r9048.json').read_bytes());proof=cpu['receipt'];assert proof['completed'];ds=proof['library_SHA256']
def rename(t):return t.replace('lookahead','fragment').replace('r9044','r9048').replace('r9042','r9048')
files=[]
for old in ['GLM_lookahead_layer_r9044.py','GLM_lookahead_layer_r9044_inside.sh','controller_GLM_lookahead_layer_r9044.py','launch_GLM_lookahead_layer_r9044.py']:
 new=rename(old);s=rename((r/old).read_text())
 if old=='GLM_lookahead_layer_r9044.py':
  s=s.replace("DSO_SHA='c1ccce0fddd3967fcda441b5db93068ed689a39c49885122124eea2498eb0077'",'DSO_SHA='+repr(ds))
  s=s.replace('Gate one-block fragment only; originalgate/down ISA bound; directPV selector alwaysOFF; QK/attention unchanged; finaltokens not qualified','Gate registerE4M3toBF16fragment plus gfx936MMAC; originalgate/down ISA bound; originalK128scaleorder; directPVfixedOFF; noWcache/globaldequantwrite; changedintrablockdot notModelqualified')
 (r/new).write_bytes(s.encode());files.append(new)
for old in ['GLM-lookahead-layer-manifest-r9044.json','GLM-lookahead-layer-root-GO-r9044.json','GLM-lookahead-layer-execution-authorized-r9044.json']:
 d=json.loads((r/old).read_bytes());new=rename(old)
 if 'sources_SHA256'in d:
  d['sources_SHA256']={n:sha(r/n)for n in files};d.update(scope='Currentlayer38B8past512 originalgate vsregisterBF16fragmentgfx936MMAC;downoriginal;oneGPUTU;profileOFF;notModelqualified',candidate_DSO_SHA256=ds,CPU_handle=proof['producer'],CPU_receipt_SHA256=cpu['receipt_SHA256']);d['DSO_SHA256'].pop('candidates/glm-gate-lookahead-r9044/libglm_gate_lookahead_r9044.so');d['DSO_SHA256']['candidates/glm-gate-fragment-r9048/libglm_gate_fragment_r9048.so']=ds;d['matrix']['candidate']='Originalgate vsregisterBF16fragmentMMAC, M1 padded16 internalonly;allread/compute intime;noexpertcache/downchange'
 elif 'DSO_SHA256'in d:d['DSO_SHA256']=ds
 (r/new).write_bytes((json.dumps(d,indent=2)+'\n').encode());files.append(new)
old='actual_GLM_lookahead_layer_r9044.py';new=rename(old);(r/new).write_bytes(rename((r/old).read_text()).encode());files.append(new)
old='observe_GLM_lookahead_layer_r9044.py';new=rename(old);(r/new).write_bytes(rename((r/old).read_text()).encode())
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_bytes())
 assert b'\r'not in(r/n).read_bytes()
print(json.dumps({'files':files,'DSO_SHA256':ds,'actualISA':proof['actual_BF16MMAC_instruction_lines'],'ISA_metadata':proof['ISA_metadata']}))
