import pathlib, hashlib, json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
items=[('candidates/glm-transient-layer-r5681/src/glm_moe_dsa_indexer_rotary.cpp','9e08792cec0fdb224201823450b9398e1eb6a474b3b90a49f386d09e2d0d594a'),('candidates/glm-transient-layer-r5681/src/glm_moe_dsa_indexer_selection.cpp','e9f2fdd7daad7b56ea51039514129a2b2661b6648667957aebf7de1cfc13951b'),('candidates/glm-M16-r8861/tree/candidates/glm-PP2-B32-r8843/tree/candidates/glm-window-model-r5797/src/glm_moe_dsa_balanced_text_model_r126.cpp','d94016e3adc4ffccac58adc24aef415c7069ad2503cb21e230f12753c0b2ad98')]
items.append(('candidates/glm-transient-layer-r5681/src/glm_moe_dsa_attention_prepare_r809.cpp','80ac6223f8fb61a9408f39172ebba773218bf936a93507b385140df96ac8d64e'))
out={'CPU_only':True,'GPU':False,'native_build':False,'sources':[]}
for rel,expect in items:
 p=r/rel;b=p.read_bytes();s=hashlib.sha256(b).hexdigest();assert s==expect,(rel,s,expect)
 out['sources'].append({'path':rel,'SHA256':s,'source':b.decode()})
print(json.dumps(out))
