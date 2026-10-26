import pathlib,json
def load():
 q=json.loads(pathlib.Path(__file__).with_name('case_plan_r9284b.json').read_bytes());validate(q);return q
def validate(q):
 if set(q)!={'schema','episodes','max_live','chunkcap','warm_prompt_tokens','warm_output_cap'}or q['schema']!=1 or q['max_live']!=16 or q['chunkcap']!=32 or q['warm_prompt_tokens']!=40 or q['warm_output_cap']!=2 or len(q['episodes'])!=4:raise ValueError('frozenplan')
 expected=[('A0-whole-row2OFF',False),('B0-whole-row2ON',True),('B1-whole-row2ON',True),('A1-whole-row2OFF',False)]
 for r,(label,mode)in zip(q['episodes'],expected):
  if set(r)!={'label','chunked','down_row2'}or r['label']!=label or type(r['chunked'])is not bool or r['chunked']is not False or type(r['down_row2'])is not bool or r['down_row2']is not mode:raise ValueError('strictbool/order')
 return q
