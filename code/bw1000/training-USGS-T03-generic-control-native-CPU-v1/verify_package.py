"""Source/AST closure only; no source dataset, processor or Torch imports."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    mf = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    assert set(mf['files']) == {p.name for p in D.iterdir() if p.is_file() and p.name != 'SOURCE_FILES.json'}
    for name, pin in mf['files'].items():
        p = D / name
        b = p.read_bytes()
        assert p.name == name and not p.is_symlink()
        assert len(b) == pin['bytes'] and hashlib.sha256(b).hexdigest() == pin['SHA256']
        if p.suffix == '.py':
            ast.parse(b, filename=name)
    plan = json.loads((D / 'PLAN.json').read_bytes())
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    assert set(plan['cases']) == {'T03.sft'}
    assert plan['native_new_denominator'] == plan['planned_train_rows'] == 45
    assert plan['planned_dev_rows'] == 0 and plan['shared_dev_rows_reused'] == 10
    assert len(json.loads((D / 'FRAMEWORK_BINDING.json').read_bytes())['files']) == 396
    assert plan['cases']['T03.sft']['metadata_files'] == 13
    assert plan['cases']['T03.sft']['native_spec']['training']['gradient_checkpointing'] is True
    assert plan['cases']['T03.sft']['attention'] == 'sdpa'
    assert claim['root_GO'] == claim['CPU_GO'] == plan['CPU_GO']
    assert claim['GPU_GO'] is False and claim['training_GO'] is False
    assert claim['budgets_seconds']['single_origin_total'] == 300
    assert claim['input']['selected_control_image_bytes'] == 7321484
    return dict(passed=True, source_members=len(mf['files']), native_new_planned=45,
                new_dev_encodes=0, shared_dev_reused=10, actual_processor=False,
                actual_Torch_or_GPU_import=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
