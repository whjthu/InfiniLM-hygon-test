"""CPU-only symbolic, synthetic group32 and actual-histogram feasibility; no GPU/import torch/weights."""
import pathlib,json,hashlib,math,random
from fractions import Fraction
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
b=D/'Kimi-INT8-feasibility-private-r9281'; b.mkdir(exist_ok=False)
read=D/'Kimi-INT8-current-source-readonly-r9281.json'; binding=json.loads(read.read_text(encoding='utf8'))
src=D/'Kimi-MoE-batch-profile-private-r9125/kimi_i192_batch_provider_r9125.cu'
assert sha(src)==binding['current_provider_source_SHA256']=='eb51cb12351b947ef2f2940ece880919ae0c5276a0292c8866d2e40693733095'
assert binding['current_provider_object_SHA256']=='f6ae35e870a8730c7f8e564d566503c6c31c1d253ea10758b5a2d3959b2f0c9d'
assert not binding['current_MD_mismatches'] and not binding['current_MD_missing']
assert all(v['matches_build'] for v in binding['current_DTK_headers'].values())
mag=[Fraction(0),Fraction(1,2),Fraction(1),Fraction(3,2),Fraction(2),Fraction(3),Fraction(4),Fraction(6)]
qmag=[0,1,2,3,4,6,8,12]
def w(code):return mag[code&7]*(-1 if code&8 else 1)
def qw(code):return qmag[code&7]*(-1 if code&8 else 1)
for code in range(16):
 for scale in range(256):
  power=Fraction(2)**(scale-127)
  assert w(code)*power==Fraction(qw(code),2)*power
# This is a rational identity, not a floating-point/no-NaN equivalence assertion.
assert 32*127*12==48768 and 48768<2**24 and 48768<2**31
rng=random.Random(9281); synth=[]
for label in ['zero','mixed','outlier']:
 a=[0.0]*192 if label=='zero' else [rng.uniform(-2,2) for _ in range(192)]
 if label=='outlier':a[0]=100
 codes=[rng.randrange(16) for _ in range(192)]; ws=[2.0**rng.randrange(-8,9) for _ in range(6)]
 got=0.; ref=0.; bound=0.; maxdot=0
 for g in range(6):
  aa=a[g*32:g*32+32]; s=max(map(abs,aa))/127 or 1.; aq=[max(-127,min(127,round(v/s))) for v in aa]
  ww=[qw(c) for c in codes[g*32:g*32+32]]; dot=sum(x*y for x,y in zip(aq,ww)); maxdot=max(maxdot,abs(dot))
  got+=dot*s*.5*ws[g]; ref+=sum(x*float(w(c))*ws[g] for x,c in zip(aa,codes[g*32:g*32+32]))
  bound+=s*.5*sum(abs(float(w(c))*ws[g]) for c in codes[g*32:g*32+32])
 assert abs(got-ref)<=bound+1e-8
 synth.append(dict(label=label,reference=ref,quantized=got,absolute_error=abs(got-ref),analytic_abs_bound=bound,max_I32_group_dot=maxdot))
t=json.loads((D/'Kimi-current-MoE-profile-private-r9264/terminal-canonical-r9264.json').read_text()); rows=[]
for s in t['shapes']:
 for o in s['owner_rows']:
  hist=o['route_histogram']; R=sum(hist); J=sum((v+15)//16 for v in hist); P=16*J; assert R==s['T']*16
  rows.append(dict(T=s['T'],owner=o['globalrank'],R=R,unique=sum(v>0 for v in hist),max_expert_rows=max(hist),J16=J,padded_rows=P,utilization=R/P,padding_factor=P/R,
   logical_INT8_OPs=2*R*192*3584,padded_INT8_OPs=2*P*192*3584,MMAC_wave_instructions=J*224*6,
   packed_weight_plus_scale_logical_grouped_reads=J*3584*102,old_perroute_logical_weight_scale_reads=R*3584*102,
   nibble_to_i8_values=J*3584*192,scale_output_elements=J*16*3584*6,
   route_partial_bytes=R*3584*4,partial_min_write_read_bytes=2*R*3584*4,
   A_I8_bytes=R*192,A_scale_F32_bytes=R*6*4,total_new_persistent_scratch_bytes=R*(3584*4+192+6*4),
   quant_min_activation_read_bytes=R*192*2,quant_min_write_bytes=R*(192+24),MMAC_min_A_gather_reads=J*224*16*192,
   MMAC_A_B_staging_bytes=J*224*6*(512+512),I32_LDS_store_read_bytes=J*224*6*256*4*2,
   warning='Logical ISA/LDS/dataflow counts, not actual HBM transactions or performance. Counts paddedzeros conservatively.'))
refs=['Kimi-current-MoE-profile-private-r9264/GPU-manifest-source.json','Kimi-current-MoE-profile-private-r9264/terminal-canonical-r9264.json','Kimi-W2-pair-private-r9280/terminal-canonical-r9280.json','InfiniLM-hygon-test/analysis/bw1000/quantized-moe-feasibility-r8912.json','integer-full-cost-reuse-r8425.json','r856-mmac-audit.json','r943-fdot2-audit.json','Kimi-cold-BF16-compact-r8669c.json','Kimi868-EP2-negative-compact-r8669zk.json','Kimi889-cyclic-EP2-final-compact-r8669zp.json','Kimi929-corrected-WFP4-real-final-r8970.json','integer-arithmetic-qualified-r8410.json','hardware_dumma_r8403.hip','matrix_hardware_roof_r4931.py','bw1000-hardware-roofline-r1311.json']
q=dict(schema=1,status='CPU_FEASIBLE_CONDITIONAL_ONE_COMPONENT_PROPOSAL_NO_BUILD_NO_GPU_GO',actions=dict(CPU=True,remote_readonly=True,GPU=False,compile=False,new_Slurm=False,weights_payload=False,Git_mutation=False),symbolic_codes_scales_checked=16*256,signed_zero='Mathematical zero preserved; signbit not preserved; exact numeric weight, not bitwise signedzero.',finite_scale_contract='Identity symbolic all byte scales; actual FP32 overflow/subnormal/scale255 semantics require actual-source guard. No payload inspected.',max_group_I32_absolute_bound=48768,synthetic_diagnostics=synth,current_provider_source_SHA256=sha(src),current_provider_object_SHA256=binding['current_provider_object_SHA256'],read_receipt_SHA256=sha(read),DTK_headers=binding['current_DTK_headers'],integer_ISA=binding['integer_build']['mmac_instruction_lines'],rows=rows,evidence_SHA256={p:sha(D/p) for p in refs},producer_SHA256=sha(__file__),limits=['No quality result: activation INT8 is lossy; rational weight identity does not imply same FP32 reduction or tokens.','Historical T64 repeats historical16 four times and routes concentrate in33experts; not true16 independentusers/online distribution.','Physical INT8 peak and exact SKU clocks UNKNOWN. 717.923TOPS empirical largeGEMM reference only.','No current benchmark/Model wall gain established.'])
(b/'CPU-feasibility-r9281.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(output=str(b/'CPU-feasibility-r9281.json'),SHA256=sha(b/'CPU-feasibility-r9281.json'),producer_SHA256=sha(__file__),routes=rows[:1]+[x for x in rows if x['T']==64][:1],symbolic_checks=q['symbolic_codes_scales_checked'])))
