import re
from quality_bench_r9100 import score_case
NUMERIC_SUFFIX='\nSolve this problem. End with #### followed by the numeric answer.'
CHOICE_SUFFIX='\nReason about the question. End with #### followed by only the correct option number (1, 2, 3, or 4).'
def prepare(q):
    task=q.get('task','gsm8k')
    if task=='gsm8k':
        text=q['question']+NUMERIC_SUFFIX
    elif task=='mmlu':
        choices=q['choices']
        if len(choices)!=4 or not all(isinstance(x,str)for x in choices):raise ValueError('four choices required')
        text=q['question']+'\n'+'\n'.join(str(i+1)+'. '+x for i,x in enumerate(choices))+CHOICE_SUFFIX
    else:raise ValueError('unsupported evaluator')
    # Gold is retained only as metadata, never composed into prompt.
    return text,{'sample_index':int(q['sample_index']),'answer':str(q['answer']),'task':task,'domain':q.get('domain','math'),'id':q.get('id',str(q['sample_index']))}

def score(q,text,stop):
    out=score_case({'id':q['id'],'gold_numeric':q['answer']},text,stop)
    return {'correct':out['strict_correct'],'extracted':out['parsed']['strict'],'numeric_diagnostics':out,'protocol':'GSM numeric generation'if q['task']=='gsm8k'else'custom MMLU numeric-index generation; not prior continuation-likelihood protocol'}
