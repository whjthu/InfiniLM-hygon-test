"""Stdlib metadata/oracle/source contract, not an operator execution result."""
import pathlib,json,ast,importlib.util,hashlib,math,sys
D=pathlib.Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('component151',D/'GPU_component_r152.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch'not in sys.modules
checks=[]
for p in [126,127,128]:
 for counts in [[1,2],[2,2]]:
  entries=m.metadata(p,counts);reverse=m.metadata(p,counts,True);assert len(entries)==sum(counts)and set(entries)==set(reverse);slots=[row*128+pos%128 for row,pos in entries];assert len(set(slots))==len(slots)and all(0<=slot<256 for slot in slots)
  touched=set()
  for slot in slots:touched.update(range(slot*576,(slot+1)*576));touched.update(range(256*576+slot*8,256*576+slot*8+7))
  assert max(touched)<((256*584+575)//576)*576
  checks.append({'position':p,'counts':counts,'packed_T':len(entries),'unique_write_slots':True,'whole_row_permutation_same_logical_inputs':True,'allowed_cache_bytes':len(touched),'passed':True})
# A fixed-input mathematical C128 boundary has no dependence on future score
# or finite background in rows overwritten by explicitly supplied score data.
def value(row,pos,j):return math.sin((row*211+pos*17+j)%997/41.)*.2
def gate(row,pos,j):return math.cos((row*197+pos*13+j)%991/37.)*.1
def oracle(row):
 result=[]
 for j in range(8):
  w=[math.exp(gate(row,p,j))for p in range(128)];result.append(sum(a*value(row,p,j)for p,a in enumerate(w))/sum(w))
 return result
for row in [0,1]:assert all(math.isfinite(x)for x in oracle(row))
for n in ['GPU_component_r152.py','CPU_build_r152.py','CPU_contract_r152.py']:ast.parse((D/n).read_bytes())
for p in D.glob('*.sh'):assert b'\r'not in p.read_bytes()
text=(D/'GPU_component_r152.py').read_text();assert 'prefix_y=compress(state,prefix)'in text and 'torch.cat([prefix_y,y],dim=0)'in text
assert "range(5)"in text and "bg=97."in text and 'delta=True'in text and 'snapshot_SHA256'in text and 'bytes_changed_outside_write_slots'in text
q={'CPU_only':True,'GPU':False,'Torch_imported':False,'Native_dlopen':False,'weights_opened':False,'source_AST_passed':True,'shells_LF':True,'metadata_cases':checks,'scalar_oracle_finite':True,'only_wrapper_HOST_compile_candidate':True,'actual_CPU_compile':'NOT_RUN','actual_GPU_component':'NOT_RUN','production_ac937_inference':'NOT_TOUCHED','F100_model_inference':'NOT_RUN_OR_REPLACED','interpretation':{'same_shape_repeat_varies':'State/output/scratch/registered-op path, not GEMM packed-shape selection (no projection GEMM present).','changed_unwritten_background_changes_valid_boundary':'Unexpected read of caller-supplied uncommitted state/payload; locate affected operator.','changed_future_at_same_T4_changes_prior_boundary':'Causality/slot alias suspicion in component, not yet model rollback proof.','cache_bytes_outside_write_mask_change':'Store ownership/overlapping byte write defect.','all_component_controls_stable':'Does not clear F100 chain; next inspect HCA prepare/rotary/cache and attention at first layer3 C128.'},'passed':True};(D/'CPU-local-contract-r152.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'cases':len(checks),'Torch_imported':False,'actual_operator_executed':False}))
