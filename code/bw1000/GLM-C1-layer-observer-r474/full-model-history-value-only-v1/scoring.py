"""Predeclared answer scoring and honest planned-denominator/partial summaries.

No tokenizer, HTTP, Native, tools or external evaluator executes in this file.
"""
from fractions import Fraction
import math,re

def parse_numeric(text):
 values=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
 try:return Fraction(values[-1].replace(',',''))if values else None
 except(ValueError,ZeroDivisionError):return None

def parse_choice(text):
 values=re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer)\s*:\s*[\[(]?([ABCD])[\])]?(?:\*\*)?(?:[.\s]|$)',text,re.IGNORECASE)
 if values:return values[-1].upper()
 match=re.fullmatch(r'\s*[\[(]?([ABCD])[\])]?\s*[.]?\s*',text,re.IGNORECASE)
 return match[1].upper()if match else None

def grade(case,content,*,natural_EOS,cap_hit,unexpected_tool=False):
 if type(content)is not str:raise TypeError('Score final public content only')
 if case['suite']=='gsm8k':
  parsed=parse_numeric(content);answer_correct=parsed is not None and parsed==Fraction(case['gold_numeric'])
 elif case['suite']=='mmlu':
  parsed=parse_choice(content);answer_correct=parsed==case['gold_choice']
 else:raise ValueError('Only the two explicitly frozen suites are supported')
 return dict(parsed_answer=str(parsed)if parsed is not None else None,parse_failure=parsed is None,answer_matches_gold=bool(answer_correct),
             correct=bool(answer_correct and natural_EOS and not cap_hit and not unexpected_tool),natural_EOS=bool(natural_EOS),cap_hit=bool(cap_hit),unexpected_tool_intent=bool(unexpected_tool),
             policy='Completion-adjusted correct requires gold, naturalEOS, no cap and no tool intent; raw answer match is also retained')

def wilson(k,n):
 if not n:return None
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return[max(0.,mid-half),min(1.,mid+half)]

def paired(data,field='correct'):
 n=len(data);gain=sum(not q[2][field]and q[4][field]for q in data);loss=sum(q[2][field]and not q[4][field]for q in data);d=gain+loss;delta=(gain-loss)/n if n else None
 p=min(1.,2*sum(math.comb(d,k)for k in range(min(gain,loss)+1))/2**d)if d else 1.
 radius=math.sqrt(2*math.log(40)/n)if n else None
 return dict(complete_pairs=n,W4_only_correct=gain,W2_only_correct=loss,accuracy_W4_minus_W2=delta,paired_Hoeffding95=[max(-1.,delta-radius),min(1.,delta+radius)]if n else None,exact_two_sided_McNemar_p=p,
             interval_scope='Observed complete pairs only; if coverage is partial this conditional interval cannot establish full planned-set quality')

def summarize(cases,rows,*,exposure_case_ids=()):
 expected={q['id']:q for q in cases}
 if len(expected)!=len(cases):raise ValueError('Duplicate planned case IDs')
 indexed={}
 for row in rows:
  key=(row['case_id'],row['window'])
  if row['case_id']not in expected or row['window']not in(2,4)or key in indexed:raise ValueError('Foreign/duplicate arm row')
  if any(type(row[k])is not bool for k in ['correct','answer_matches_gold','natural_EOS','cap_hit','parse_failure']):raise ValueError('Typed score flags required')
  indexed[key]=row
 exposure=set(exposure_case_ids)
 if not exposure<=set(expected):raise ValueError('Exposure classification must come from all frozen actual encodings')
 def view(ids):
  ids=list(ids);n=len(ids);arms={}
  for w in(2,4):
   known=[indexed[(i,w)]for i in ids if(i,w)in indexed];k=sum(q['correct']for q in known);missing=n-len(known)
   arms[str(w)]=dict(planned=n,observed=len(known),correct=k,raw_answer_matches_gold=sum(q['answer_matches_gold']for q in known),raw_planned_answer_accuracy=sum(q['answer_matches_gold']for q in known)/n if n else None,uncompleted_or_not_attempted=missing,
      planned_completion_adjusted_accuracy=k/n if n else None,planned_score_Wilson95=wilson(k,n),
      missing_outcome_accuracy_bounds=[k/n,(k+missing)/n]if n else None,
      natural_EOS=sum(q['natural_EOS']for q in known),caps=sum(q['cap_hit']for q in known),parse_failures=sum(q['parse_failure']for q in known),wrong_or_invalid_observed=len(known)-k)
  complete=[{2:indexed[(i,2)],4:indexed[(i,4)]}for i in ids if(i,2)in indexed and(i,4)in indexed]
  lower=upper=0
  for i in ids:
   a=indexed.get((i,2));b=indexed.get((i,4));a0=a1=int(a['correct'])if a is not None else 0;b0=b1=int(b['correct'])if b is not None else 0
   if a is None:a1=1
   if b is None:b1=1
   lower+=b0-a1;upper+=b1-a0
  return dict(arms=arms,expected_pairs=n,pair_coverage=len(complete)/n if n else None,observed_complete_pairs=paired(complete),observed_complete_pair_raw_answer_score=paired(complete,'answer_matches_gold'),
              all_planned_quality_difference_missing_outcome_bounds=[lower/n,upper/n]if n else None,
              planned_denominator_never_shrinks=True)
 by_suite={s:view(i for i,c in expected.items()if c['suite']==s)for s in('gsm8k','mmlu')}
 return dict(per_suite=by_suite,pooled=view(expected),actual_multigroup_exposure_subset=view(i for i in expected if i in exposure),
             exposure_scope='Classified before outputs using actual CPU prompt lengths and confirmed by live callbacks; never answer-based filtering. Report unexposed questions in full planned denominator.',
             noninferiority_proven=False,noninferiority_margin_prespecified=None,full_benchmark=False,automatic_all_correct_gate=False,
             limits='Explicit128+128 custom chat few-shot screen, not full benchmark or lm-eval reproduction. Wilson refers to completion-adjusted planned scores; missing-outcome bounds and observed-pair CI are shown separately. Quality wrong/cap/parse failures do not authorize stopping or dropping cases.')
