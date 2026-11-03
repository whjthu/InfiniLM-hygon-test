"""Finite grader/partial-plan CPU leaves only; no benchmark model or HTTP runs."""
from pathlib import Path
import importlib.util,json
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r444_actual_score',D/'scoring.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def check():
 g=dict(id='unit-g',suite='gsm8k',gold_numeric='42');a=dict(id='unit-m1',suite='mmlu',gold_choice='B');b=dict(id='unit-m2',suite='mmlu',gold_choice='C');checks=0
 for text,want in [('Answer: B','B'),('**Answer: (C)**','C'),('D','D'),('Reason A or B.\nFinal answer: A','A'),('The answer is not in the requested format',None)]:assert S.parse_choice(text)==want;checks+=1
 assert S.parse_numeric('work\n#### 1,200')==1200;checks+=1
 def row(case,w,text,cap=False):return dict(case_id=case['id'],window=w,**S.grade(case,text,natural_EOS=not cap,cap_hit=cap))
 rows=[row(g,2,'#### 42'),row(g,4,'#### 43'),row(a,2,'Answer: B',True),row(a,4,'unparsed')]
 q=S.summarize([g,a,b],rows,exposure_case_ids=[g['id']]);p=q['pooled'];assert p['expected_pairs']==3 and p['arms']['2']['planned']==3 and p['arms']['2']['correct']==1 and p['arms']['4']['correct']==0
 assert p['arms']['2']['caps']==1 and p['arms']['4']['parse_failures']==1 and p['arms']['2']['uncompleted_or_not_attempted']==1 and p['arms']['4']['uncompleted_or_not_attempted']==1
 assert p['arms']['2']['raw_answer_matches_gold']==2 and p['observed_complete_pairs']['accuracy_W4_minus_W2']==-.5 and p['all_planned_quality_difference_missing_outcome_bounds']==[-2/3,0]
 assert q['actual_multigroup_exposure_subset']['expected_pairs']==1 and not q['noninferiority_proven']and not q['automatic_all_correct_gate'];checks+=1
 for bad in[rows+[rows[0]],[dict(rows[0],case_id='foreign')],[dict(rows[0],window=3)]]:
  try:S.summarize([g,a,b],bad)
  except ValueError:pass
  else:raise AssertionError('Foreign/duplicate score row accepted')
  checks+=1
 empty=S.summarize([g,a,b],[]);assert empty['pooled']['arms']['2']['planned']==3 and empty['pooled']['all_planned_quality_difference_missing_outcome_bounds']==[-1,1]and empty['pooled']['pair_coverage']==0;checks+=1
 return dict(passed=True,checks=checks,wrong_cap_parse_and_uncompleted_remain_in_planned_denominator=True,raw_answer_score_and_completion_adjusted_score_separate=True,partial_uncertainty_not_zeroed=True,noninferiority_proven=False,recording_leaves=['three unit labels','content text','EOS/cap flags'],actual_benchmark_questions_scored=False,HTTP=0,Native=0,GPU=0,Slurm=0,Model=0)
if __name__=='__main__':print(json.dumps(check(),indent=2))
