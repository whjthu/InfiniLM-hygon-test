from pathlib import Path
import json,hashlib,base64,datetime,math,statistics,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');refs=json.loads(base64.b64decode('W3sicGF0aCI6ICIvcHVibGljL2hvbWUvYWNsc3lscWlkZi9oeWdvbi1pbmZlcmVuY2UtcmVmYWN0b3ItMjAyNjEwMTQvcmVzdWx0cy9LaW1pLWxvbmctcGFpci1yMzQ3LWxvbmdwYWlyLTIwMjYxMTAyLWQtd2F0Y2guanNvbiIsICJTSEEyNTYiOiAiMDgyOTcwNjExNTM5NzJlODgyMjBlM2U5ZTgzY2Y1NmIxY2Q4MDhlZWNhMjlkNjcyYzBkZjhmOGVjOWMwMGZhMCJ9LCB7InBhdGgiOiAiL3B1YmxpYy9ob21lL2FjbHN5bHFpZGYvaHlnb24taW5mZXJlbmNlLXJlZmFjdG9yLTIwMjYxMDE0L3Jlc3VsdHMvS2ltaS1sb25nLXBhaXItcjM0Ni1sb25ncGFpci0yMDI2MTEwMi1kLW5vZGUwLmpzb24iLCAiU0hBMjU2IjogImRiNDRiODk1YjM4NmJkNmQzNzA5NGYzZjkwYjAzYjEzYTdkODI5OGFkZmUxYTlmMjU4YjMxYWVjZDVhZjNmZjcifSwgeyJwYXRoIjogIi9wdWJsaWMvaG9tZS9hY2xzeWxxaWRmL2h5Z29uLWluZmVyZW5jZS1yZWZhY3Rvci0yMDI2MTAxNC9yZXN1bHRzL0tpbWktbG9uZy1wYWlyLXIzNDYtbG9uZ3BhaXItMjAyNjExMDItZC1ub2RlMS5qc29uIiwgIlNIQTI1NiI6ICI3NzRkNzZiNzhjOTQwNWJkYTExMmUyNDQ0MzUxY2QxYzNlMDQyMGMzM2FiNmQyMjYyM2UxY2Q5NTNiZjU5MDE1In0sIHsicGF0aCI6ICIvcHVibGljL2hvbWUvYWNsc3lscWlkZi9oeWdvbi1pbmZlcmVuY2UtcmVmYWN0b3ItMjAyNjEwMTQvcmVzdWx0cy9LaW1pLWxvbmctcGFpci1yMzQ2LWxvbmdwYWlyLTIwMjYxMTAyLWQtbm9kZTIuanNvbiIsICJTSEEyNTYiOiAiNTM4OTYxYWRkMTgxZjJlNzA2OWUwYThiNWViOTljZmUyNTg1NjE0YmM3NDQ5MjhjZTEzZGFhZmQxYWNlMjVjYiJ9LCB7InBhdGgiOiAiL3B1YmxpYy9ob21lL2FjbHN5bHFpZGYvaHlnb24taW5mZXJlbmNlLXJlZmFjdG9yLTIwMjYxMDE0L3Jlc3VsdHMvS2ltaS1sb25nLXBhaXItcjM0Ni1sb25ncGFpci0yMDI2MTEwMi1kLW5vZGUzLmpzb24iLCAiU0hBMjU2IjogIjllMmEyZmRhZjI2OTc5ZmNkYTk0ZTc4N2NhMmNmMzZhMmRmOWU4ODRiMTAxMWI3MWVkNGJiOWFjMjRkMTBlNTQifSwgeyJwYXRoIjogIi9wdWJsaWMvaG9tZS9hY2xzeWxxaWRmL2h5Z29uLWluZmVyZW5jZS1yZWZhY3Rvci0yMDI2MTAxNC9yZXN1bHRzL0tpbWktbG9uZy1kcmFmdC1yMzM0LWxvbmdwYWlyLTIwMjYxMTAyLWQuanNvbiIsICJTSEEyNTYiOiAiODViZTI2M2IzMTJiNWIzODU1NDQ0NGRmZmI2Y2U5NGI2NjY3YTc1ODAyMDZiMmI3MWJlYjU1YjRhYmMxMGYyZCJ9XQ=='));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports=[]
for v in refs:
 p=Path(v['path']);assert sha(p)==v['SHA256'];reports.append(json.loads(p.read_bytes()))
watch=reports[0];assert watch['passed']is False and watch['error_type']=='AssertionError'and 'post'in str(watch['error_frames'])
leaders=reports[1:5];draft=reports[5];assert {q['node']for q in leaders}==set(range(4))
driver=R/'incoming/Kimi-long-pair-driver-r337';assert sha(driver/'SOURCE_MANIFEST.json')=='445f96b37c1ceab9e62c3599879834b8be21da9b9585384c942c12ffef86644f'
m=json.loads((driver/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(driver/n)==h for n,h in m['files_SHA256'].items())
frozen=json.loads((driver/'frozen_inputs.json').read_bytes());fixtures={f['name']:f for f in frozen['fixtures']}
model=Path('/public/home/aclsylqidf/shared/Kimi-K3');assert sha(model/'tokenizer_config.json')==frozen['tokenizer']['tokenizer_config_SHA256'];assert sha(model/'tiktoken.model')==frozen['tokenizer']['tiktoken_model_SHA256']
cfg=json.loads((model/'tokenizer_config.json').read_bytes());mapping={}
for line in (model/'tiktoken.model').read_bytes().splitlines():
 token,rank=line.split();mapping[int(rank)]=base64.b64decode(token)
assert len(mapping)==163584
for i in range(163584,163840):mapping[i]=cfg['added_tokens_decoder'].get(str(i),{}).get('content','<|reserved_token_%d|>'%i).encode()
def decode(ids):return b''.join(mapping[i]for i in ids).decode('utf8')
sig=None
for q in leaders:
 assert q['completed']and q['all8_owned_workers_shutdown_completed']and q['capture_CPU_cleanup_completed']and q['control_CPU_cleanup_completed']and not q['cleanup_errors']
 assert q['source_manifest_SHA256']=='909cea03171e52e81f377d1f2d61425a7d46778522fb28c4be989b46f41436b1'and q['Native_SHA256']=='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
 assert q['first_capture_validation']['passed']and q['first_capture_validation']['layerIDs']==[19,37,66,78,90]
 pair=q['pair_report'];assert pair['completed']and pair['normal_draft_protocol_close']and len(pair['requests'])==4
 current=[]
 for i,r in enumerate(pair['requests']):
  ids=r['generated_ids'];assert r['ordinal']==i and r['tokens']==len(ids)==59 and ids[-1]in[163585,163586]and not set(ids[:-1])&{163585,163586}
  assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==r['ids_SHA256']
  assert r['finish']=='eos'and r['quality']['passed']and r['normal_reset']
  fixture=fixtures[r['name']];assert r['prompt_SHA256']==fixture['prompt_ID_SHA256']and r['input_tokens']==len(fixture['prompt_tokens'])
  assert r['quality']['details']['arguments']==fixture['expected_calculate_arguments']
  current.append((r['prompt_SHA256'],r['ids_SHA256'],r['tokens'],r['finish']))
 if sig is None:sig=current
 assert current==sig
assert draft['completed']and draft['passed']and draft['normal_protocol_close']and draft['source_manifest_SHA256']=='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31'
assert draft['phase_counts']==dict(begin=2,append_prompt=81,seal=2,query=16,append_committed=14,retire=2,close=1)and draft['commands']==118
assert draft['allocator_trim_count']==1 and all(v['free_after_bytes']>=4*2**30 for v in draft['memory_fences'])
node0=next(q for q in leaders if q['node']==0);arms=[]
def pct(xs,p):
 xs=sorted(xs);x=(len(xs)-1)*p;i=int(x);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)
for r in node0['pair_report']['requests']:
 events=r['emitter_events'];flatten=[v for e in events for v in e['tokens']];assert flatten==r['generated_ids']
 ticks=[e['at_s']for e in events for token in e['tokens']if token not in[163585,163586]]
 assert len(ticks)==58 and ticks==sorted(ticks)and all(math.isfinite(x)and x>=0 for x in ticks)
 gaps=[b-a for a,b in zip(ticks,ticks[1:])];rate=57/(ticks[-1]-ticks[0]);assert abs(rate-r['backend_decode_tok_s'])<1e-10
 proposals=sum(sum(c['width'])-len(c['width'])for c in r['Native_calls']if c['kind']=='verify')
 # Keep exact individual cycle schema for the coordinator; no inferred acceptance.
 arms.append(dict(ordinal=r['ordinal'],name=r['name'],mode=r['mode'],input_tokens=r['input_tokens'],output_tokens=59,useful_output_tokens=58,TTFT_s=ticks[0],first_last_s=ticks[-1]-ticks[0],backend_decode_tok_s=rate,ITL_max_s=max(gaps),ITL_P95_s=pct(gaps,.95),ITL_P99_s=pct(gaps,.99),generation_local_wall_s=r['generation_local_wall_s'],request_through_reset_s=r['request_through_reset_local_wall_s'],full33_capacity_generation_tok_per_card_s=58/r['generation_local_wall_s']/33,decoded_output=decode(r['generated_ids'][:-1]),expected_arguments=fixtures[r['name']]['expected_calculate_arguments'],proposal_count=proposals,cycles=r['cycles']))
raw=subprocess.check_output(['sacct','-S','2026-11-01','-j','910380.550,910371.672,910407.719','-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed'],text=True,timeout=25)
assert '910407.719|FAILED|1:0|'in raw and '910380.550|COMPLETED|0:0|'in raw and '910371.672|COMPLETED|0:0|'in raw
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,root_original6file_SHA_and4leader_signatures_and_raw_emitter_audit=True,refs=refs,arms=arms,draft_phase_counts=draft['phase_counts'],one_time_draft_connect_load_before_request_start_s=node0['pair_report']['draft_connect_load_handshake_local_s'],accounting=raw,strict_XTML_format_evidence='Frozen original grader passed; coordinator independently decodes actual official ranks below and verifies semantic integer arguments',scope='Same older327DSO AR/spec engine4arm diagnostic; no HTTP/online/latest99/stableABBA or broadbenchmark qualification',raw_features_exported=False)))
