"""CPU read-only summary of retained exact HIP compiler assembly; no execution."""
from pathlib import Path
import json,re,sys,hashlib,subprocess
B=Path('/refactor/candidates/Kimi-MoE-phase-component-r161');q=json.loads((B/'CPU-build.json').read_bytes());assert q['completed']and q['CPU_only']and not q['GPU']
rows=[]
for f in q['retained_ISA']:
 p=Path(f['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==f['SHA256'];text=raw.decode(errors='replace')
 if p.suffix!='.s':continue
 for match in re.finditer(r'(?m)^\s*\.globl\s+(\S*groupedW13I192R8669y\S*)',text):
  name=match.group(1);start=match.start();tail=text[start:];n=re.search(r'(?m)^\s*\.size\s+'+re.escape(name)+r'\b',tail);body=tail[:n.start()]if n else tail[:60000]
  demangle=subprocess.check_output(['c++filt',name],text=True).strip();assert len(body)<200000
  counts={k:len(re.findall(r'(?m)^\s*'+k+r'\b',body))for k in ['v_ldexp_f32','v_ldexp_f32_e32','v_ldexp_f32_e64','global_load_ubyte','global_load_dword','s_load_dword','v_bfi_b32','v_lshlrev_b32','v_cndmask_b32_e32','v_cndmask_b32_e64']}
  meta=tail[:120000];fields={k:re.findall(r'\.amdhsa_'+k+r'\s+(\d+)',meta)[:1]for k in ['next_free_vgpr','next_free_sgpr','group_segment_fixed_size','private_segment_fixed_size']}
  rows.append({'symbol':name,'demangle':demangle,'assembly_file_SHA256':f['SHA256'],'body_SHA256':hashlib.sha256(body.encode()).hexdigest(),'opcode_counts_static':counts,'metadata_first_match_diagnostic':fields,'contains_direct':bool(re.search(r', true>',demangle)),'context_excerpt':body[:3000]})
assert len(rows)==2,('need_exact2templatekernels',len(rows));assert sum(x['contains_direct']for x in rows)==1
out=B/'CPU-ISA-review-r161.json';assert not out.exists();out.write_text(json.dumps({'CPU_only':True,'GPU':False,'rows':rows,'interpretation':'Counts arestaticinstructionoccurrences,notdynamicexecutions/cycles; metadata firstmatchmustbeverifiedagainstexactkerneldescriptor. Rootreviewrequired,noautoGPU.'},indent=2)+'\n');print(json.dumps({'CPU_ISA_summary':True,'kernel_count':len(rows),'GPU':False}))
