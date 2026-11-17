"""Static local package closure only, no actual processor/model/data reads."""
import ast
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent


def main():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes());actual={p.name for p in D.iterdir() if p.is_file() and p.name!='SOURCE_FILES.json'}
    assert set(mf['files'])==actual
    for name,pin in mf['files'].items():
        p=D/name;b=p.read_bytes();assert not p.is_symlink() and len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
        if p.suffix=='.py':ast.parse(b,filename=name)
    plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    assert set(plan['cases'])=={'T04.sft'} and plan['native_new_denominator']==45 and plan['planned_dev_rows']==0 and plan['shared_dev_rows_reused']==10
    assert len(json.loads((D/'FRAMEWORK_BINDING.json').read_bytes())['files'])==396 and plan['cases']['T04.sft']['metadata_files']==14
    assert claim['budgets_seconds']['single_origin_total']==300 and claim['budgets_seconds']['native_work_deadline_offset']==260
    assert claim['input']['selected_control_image_bytes']==7321484 and claim['input']['source_caption_body_exports_to_PC']==0
    return dict(passed=True,source_files=len(actual),new_native_planned=45,shared_dev_reused=10,actual_processor=False,actual_Torch_or_GPU_import=False)

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
