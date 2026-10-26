import pathlib,json
def load():
 q=json.loads(pathlib.Path(__file__).with_name('case_plan_r9260.json').read_bytes());validate(q);return q
def validate(q):
 if set(q)!={'schema','episodes','max_live','chunkcap','warm_prompt_tokens','warm_output_cap'}or q['schema']!=1 or q['max_live']!=16 or q['chunkcap']!=32 or q['warm_prompt_tokens']!=40 or q['warm_output_cap']!=2 or len(q['episodes'])!=4:raise ValueError('frozenplan')
 for r,(label,mode)in zip(q['episodes'],[('A0-whole',False),('B0-chunk32',True),('B1-chunk32',True),('A1-whole',False)]):
  if set(r)!={'label','chunked'}or r['label']!=label or type(r['chunked'])is not bool or r['chunked']is not mode:raise ValueError('strictbool/order')
 return q
