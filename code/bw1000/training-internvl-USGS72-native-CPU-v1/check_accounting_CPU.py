"""New accounting schema/count boundaries; no query, process or native call."""
import json
from pathlib import Path
from accounting_contract import select_original,actual_step_count


def main():
    parent='1087656';label='training-internvl-USGS72-native-CPU-v1';node='a14r3n04'
    allocation=f'{parent}|allocation|RUNNING|0:0|100|a14r3n[03-04]|cpu=256,node=2'
    step=f'{parent}.999|{label}|COMPLETED|0:0|20|{node}|cpu=8,mem=12G,node=1'
    r=select_original([allocation,''],parent,label,node)
    assert r['parent'][0]==parent and r['step'] is None and actual_step_count(None,parent,node) is None
    r=select_original([allocation,step],parent,label,node)
    assert actual_step_count(r['step'],parent,node)==1
    rejected=[]
    cases={
        'malformed_unrelated_row_not_silently_dropped':[allocation,'other|only|three'],
        'missing_parent':[step],
        'duplicate_parent':[allocation,allocation],
        'duplicate_original_label':[allocation,step,step],
        'wrong_parent_step':[allocation,step.replace(parent+'.999','777.999')],
        'nonnumeric_step':[allocation,step.replace(parent+'.999',parent+'.extern')],
        'wrong_node':[allocation,step.replace(node,'other-node')],
    }
    for name,lines in cases.items():
        try:select_original(lines,parent,label,node)
        except ValueError:rejected.append(name)
        else:raise AssertionError('invalid accounting accepted: '+name)
    out=Path(__file__).resolve().parents[3]/'local-only/training-internvl-USGS72-native-CPU-v1-accounting-checks'
    out.mkdir(exist_ok=False)
    result=dict(passed=True,normal_cases=2,rejected_cases=rejected,absent_actual_step_count=None,
        absent_actual_step_state='UNKNOWN_NOT_COUNT1',synthetic_accounting_only=True,
        old_checks_rerun=False,actual_Slurm_queries=0,actual_submissions=0,Torch_imported=False,GPU=0)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=True,normal_cases=2,rejected_cases=len(rejected),absent_actual_step_count=None,Slurm=0),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
