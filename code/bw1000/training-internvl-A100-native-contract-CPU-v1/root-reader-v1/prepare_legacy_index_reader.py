"""Fix only the independent reader's observed legacy index-name assumption."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;B=D.parent
first=json.loads((D/'root-reader-result.json').read_bytes());assert first['transport_rc']==1
index=json.loads((B/'training-internvl-original-contract-metadata-v1/original-source/index.json').read_bytes())['weight_map']
observed={'model.vision_tower':'vision_tower','model.multi_modal_projector':'multi_modal_projector'}
cpu=json.loads((D/'transport-outcome.json').read_bytes())['value']['CPU_result']
for actual,legacy in observed.items():
 assert not any(name.startswith(actual+'.') for name in index)
 assert cpu['frozen'][actual]['tensors']==sum(name.startswith(legacy+'.') for name in index)
text=(D/'read_original_terminal.py').read_text(encoding='utf8')
old="assert cpu['frozen'][path]['tensors']==sum(name.startswith(path+'.') for name in index) and cpu['frozen'][path]['parameters']>0"
new="legacy={'model.vision_tower':'vision_tower','model.multi_modal_projector':'multi_modal_projector'}[path]\n assert not any(name.startswith(path+'.') for name in index)\n assert cpu['frozen'][path]['tensors']==sum(name.startswith(legacy+'.') for name in index) and cpu['frozen'][path]['parameters']>0"
assert old in text;text=text.replace(old,new)
text=text.replace('frozen_tensor_counts_match_original_index=True','frozen_tensor_counts_match_original_index_under_legacy_names=True,checkpoint_value_name_conversion_qualified=False')
for before,after in [('root-reader-intent.json','root-reader2-intent.json'),('root-reader.stdout.private.json','root-reader2.stdout.private.json'),('root-reader.stderr.private.txt','root-reader2.stderr.private.txt'),('root-reader-result.json','root-reader2-result.json')]:text=text.replace(before,after)
ast.parse(text);(D/'read_original_terminal_legacy_index_v2.py').write_bytes(text.encode())
diagnosis=dict(original_reader_rc=1,original_reader_result_SHA256=hashlib.sha256((D/'root-reader-result.json').read_bytes()).hexdigest(),original_reader_stderr_SHA256=hashlib.sha256((D/'root-reader.stderr.private.txt').read_bytes()).hexdigest(),
 actual_original_index_name_mapping=observed,frozen_tensor_counts={k:v['tensors'] for k,v in cpu['frozen'].items()},source_run_repeated=False,model_weight_values_read=False,
 reader_fix='Only use observed legacy index names for independent count correspondence; preserve actual native HF module paths and frozen boundary contract.',checkpoint_value_name_conversion_qualified=False)
(D/'legacy-index-reader-diagnosis.json').write_bytes((json.dumps(diagnosis,indent=2)+'\n').encode())
print(json.dumps(dict(prepared=True,frozen_counts=diagnosis['frozen_tensor_counts'],source_run_repeated=False,checkpoint_value_name_conversion_qualified=False)))
