"""CPU source/shape/unique-storage alias contract; no tensor library or Runtime."""
from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent
S=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/refactor/incoming/Kimi-TP32-I96-template-component-r163')
pins=json.loads((D/'parent-pins-r167.json').read_bytes());old=(S/'kimi_moe_I96_r163.cpp').read_bytes();new=(D/'kimi_moe_I96_exact_r167.cpp').read_bytes()
assert hashlib.sha256(old).hexdigest()==pins['parent_source_SHA256']
begin=old.index(b'void KimiK3Experts::register_mxfp4_experts');end=old.index(b'const KimiK3Mxfp4MoeWeights &KimiK3Experts::mxfp4_weights()',begin)
section=old[begin:end];assert section.count(b'Tensor::empty(')==4 and section.count(b'Tensor::empty_parameter(')==0
expected=old[:begin]+section.replace(b'Tensor::empty(',b'Tensor::empty_parameter(')+old[end:]
assert new==expected,'only four allocator API replacements permitted'
assert new[:begin]==old[:begin] and new[begin+len(section)+4*len(b'_parameter'):]==old[end:]
assert b'narrow({{0, expert, 1}, {1, row_start, row_count}})'in section and b'->squeeze(0)'in section
assert b'infinilm::weights::Weight(expert_view, tp_dim, tp_rank_, tp_size_)'in section
# Byte-level unique storage geometry; leaves are views, not independent mallocs.
E,H,I=896,3584,96
shapes={'w13':[E,2*I,H//2],'s13':[E,2*I,H//32],'w2':[E,H,I//2],'s2':[E,H,I//32]}
sizes={n:__import__('math').prod(s)for n,s in shapes.items()}
assert sizes=={'w13':308281344,'s13':19267584,'w2':154140672,'s2':9633792}
for expert in range(E):
 for bank,row_width in [('w13',H//2),('s13',H//32)]:
  base=expert*2*I*row_width;middle=base+I*row_width;last=middle+I*row_width
  assert base<middle<last<=sizes[bank]
  if expert+1<E:assert last==(expert+1)*2*I*row_width
 for bank,row_width in [('w2',I//2),('s2',I//32)]:
  base=expert*H*row_width;last=base+H*row_width;assert 0<=base<last<=sizes[bank]
  if expert+1<E:assert last==(expert+1)*H*row_width
assert sum(sizes.values())==491323392
regular={'w13':308281344,'s13':33554432,'w2':268435456,'s2':16777216}
padding=sum(regular.values())-sum(sizes.values());assert padding==135725056
print(json.dumps({'passed':True,'CPU_only':True,'Runtime_created':False,'tensor_library_imported':False,'GPU':False,'four_call_only_diff':True,'same_shape_stride_alias_tpdim_load_source':True,'expert_leaf_views':E*6,'unique_packed_storages':4,'packed_requested_bytes':sizes,'alignment256_no_extra':all(n%256==0 for n in sizes.values()),'removed_regular_pool_class_padding_per_MoE_design_bytes':padding,'removed_all92_design_bytes':92*padding,'actual_GPU_saving':'UNKNOWN','fullModel_peak':'UNKNOWN_STOP'}))
