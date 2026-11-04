"""Frozen CPU945 first16GSM+first16MMLU, no encode or outcome-dependent selection."""
from pathlib import Path
import argparse,hashlib,json
H=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
ORDINALS=list(range(16))+list(range(128,144))
def build(encoding,gold):
 raw=Path(encoding).read_bytes();assert H(raw)=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a';e=json.loads(raw)
 raw=Path(gold).read_bytes();assert H(raw)=='c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38';g={v['id']:v for v in json.loads(raw)['cases']};records=[]
 for ordinal in ORDINALS:
  meta=e['cases'][ordinal];b=Path(meta['private_path']).read_bytes();assert H(b)==meta['private_SHA256'];v=json.loads(b)
  assert meta['selected_ordinal']==ordinal and v['case_id']==meta['case_id']and H(compact(v['body']))==meta['request_body_SHA256']and H(compact(v['prompt_token_ids']))==meta['prompt_IDs_SHA256']
  assert meta['output_budget']==v['body']['max_tokens']==4096 and len(v['prompt_token_ids'])+4096<=32768
  assert v['body']['chat_template_kwargs']['enable_thinking']is False and H(compact(g[v['case_id']]))==meta['source_case_SHA256']
  records.append(dict(case_id=v['case_id'],ordinal=ordinal,body=v['body'],body_SHA256=meta['request_body_SHA256'],prompt_token_ids=v['prompt_token_ids'],prompt_IDs_SHA256=meta['prompt_IDs_SHA256'],gold=g[v['case_id']]))
 return dict(schema=1,records=records,ordinals=ORDINALS,cases=32,HTTP=64,encoding_CPU='910407.945',encoding_SHA256=H(Path(encoding).read_bytes()),new_encode=False,selection='First16eachsuite; fixed before outputs',pair_order='alternating serial/cohort then cohort/serial by fourcase group',actual_processor_reused_CPU945=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--encoding',required=True);p.add_argument('--gold',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=build(a.encoding,a.gold)
 with Path(a.output).open('x')as f:json.dump(q,f,ensure_ascii=False);f.write('\n')
 print(json.dumps(dict(cases=32,HTTP=64,output_SHA256=H(Path(a.output).read_bytes()),new_encode=False)))
if __name__=='__main__':main()
