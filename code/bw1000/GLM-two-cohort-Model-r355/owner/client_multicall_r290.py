"""Source-only bounded real multicall supplement. Never imported by r223.

No launcher or GPU authority is included. Wire traces and tool data stay local.
Two modes, at most two actual HTTP requests each, no retries or fabricated calls.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import secrets
import time

from source_pin_r290 import load_wire,profiles
from client_owner_r290 import bind_identity,bound_report,row as completed_row
PATHS = ('left.json', 'right.json')
TOOLS = [{'type': 'function', 'function': {'name': 'read_file',
    'description': 'Read one of the two actual JSON data files.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'enum': list(PATHS)}},
                   'required': ['path'], 'additionalProperties': False}}}]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_calls(message, finish):
    calls = message.get('tool_calls', [])
    if finish != 'tool_calls' or len(calls) != 2:
        raise ValueError('MULTICALL_NOT_OBSERVED: exactly two requested real calls required')
    seen, paths = set(), []
    for call in calls:
        call_id = call.get('id')
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ValueError('Missing or duplicate external opaque ID')
        seen.add(call_id)
        if call.get('type') != 'function' or call['function']['name'] != 'read_file':
            raise ValueError('Unexpected generated function')
        args = json.loads(call['function']['arguments'])
        if not isinstance(args, dict) or set(args) != {'path'} or args['path'] not in PATHS:
            raise ValueError('Unexpected generated path or arguments')
        paths.append(args['path'])
    if set(paths) != set(PATHS):
        raise ValueError('Both actual files must be requested once')
    return list(zip(calls, paths))


def execute_and_reverse(validated, workspace):
    results, expected = [], {}
    for call, name in validated:
        data = json.loads((workspace / name).read_bytes())
        values = {row['id']: row['value'] for row in data['records']}
        total = sum(values[key] for key in data['query_ids'])
        expected[name] = {'nonce': data['nonce'], 'sum': total}
        results.append({'role': 'tool', 'tool_call_id': call['id'], 'content': compact(data)})
    expected['total'] = sum(expected[name]['sum'] for name in PATHS)
    return list(reversed(results)), expected


def check_final(message, finish, expected):
    if finish != 'stop' or message.get('tool_calls'):
        raise ValueError('Final answer must naturally stop without tool calls')
    actual = json.loads(message.get('content') or '')
    if actual != expected:
        raise ValueError('Incorrect file/nonce/sum association or total')
    # JSON booleans compare equal to small ints in Python; reject that ambiguity.
    if type(actual.get('total')) is not int or any(type(actual[name].get('sum')) is not int for name in PATHS):
        raise ValueError('Final sums must be integers')


def response_binding(folder, serial, usage, messages, thinking):
    wire = json.loads((folder/('request-%03d.wire' % serial)).read_bytes())
    request_id = wire.get('id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('Missing actual HTTP response request ID')
    rc = [hashlib.sha256((v.get('reasoning_content') or v.get('reasoning')).encode()).hexdigest()
          for v in messages if v.get('role') == 'assistant' and (v.get('reasoning_content') or v.get('reasoning'))]
    return {'request_id': request_id, 'usage': usage, 'RC_SHA256_sequence': rc, 'thinking': thinking}


def owner_receipts(path,args,session,bindings,budget,remaining):
 safe=[]
 for offset,b in enumerate(bindings):
  index=34+(2 if b['thinking']else 0)+offset;r=completed_row(path,args,b['request_id'],index,session,budget,remaining);h=r['admission']['history_render_audit'];assert r['native_finish_reason']=='eos_token'and r['prompt_tokens']==b['usage']['prompt_tokens'];assert h['source_RC_SHA256_sequence']==b['RC_SHA256_sequence']and h['generation_thinking']is True and h['pending_thinking_prefix_open']is True and h['generation_prefix_correct']and h['RC_one_contribution_per_source_field']and h['RC_order_preserved']and h['RC_contributions_nonoverlapping']and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']
  safe.append({'admitted_index':index,'prompt_tokens':r['prompt_tokens'],'output_budget':budget,'natural_EOS':True,'normal_reset':True,'actual_prompt_SHA256':h['actual_prompt_SHA256'],'Native_prompt_token_ids_SHA256':h['Native_prompt_token_ids_SHA256']})
 return safe


def run_mode(p0, args, root, profile_data, thinking, remaining):
    label, budget = ('thinking', 8192) if thinking else ('chat', 4096)
    folder = root / label
    folder.mkdir()
    workspace = folder / 'workspace'
    workspace.mkdir()
    records = profile_data['records']
    middle = len(records) // 2
    for name, rows in zip(PATHS, (records[:middle], records[middle:])):
        if len(rows) < 4:
            raise ValueError('Insufficient real source records')
        data = {'nonce': secrets.token_hex(16), 'query_ids': [rows[i]['id'] for i in (0, len(rows)//3, 2*len(rows)//3, len(rows)-1)], 'records': rows}
        with (workspace / name).open('xb') as stream:
            stream.write((compact(data)+'\n').encode())
    session = 'P1-multicall-' + label + '-GLM290-' + secrets.token_hex(16)
    client = p0.WireClient(args.base_url, args.model, folder, budget, min(args.timeout, remaining()-30))
    client.session = session
    messages = [{'role': 'system', 'content': 'Use real tools and preserve their actual results. Tool result order may differ from call order; associate each result by its tool_call_id.'},
        {'role': 'user', 'content': 'In ONE response call read_file twice, once for left.json and once for right.json. After both results, find each file query_ids in its records and sum those four values. Output ONLY a JSON object with keys left.json and right.json, each containing its actual nonce and integer sum, and total containing both sums added together. Do not guess file contents.'}]
    usages, events, bindings = [], [], []
    original = copy.deepcopy(messages)
    message, finish, usage, event = client.request(messages, 'required', False, TOOLS, max_tokens=budget, thinking=thinking, parallel=True)
    remaining()
    if messages != original or event['HTTP_status'] != 200:
        raise ValueError('First actual HTTP request failed or history mutated')
    usages.append(usage); events.append(event)
    bindings.append(response_binding(folder, 1, usage, messages, thinking))
    validated = validated_calls(message, finish)
    reversed_results, expected = execute_and_reverse(validated, workspace)
    messages.append(copy.deepcopy(message))
    messages.extend(reversed_results)
    original = copy.deepcopy(messages)
    client.timeout = min(args.timeout, remaining()-30)
    message, finish, usage, event = client.request(messages, 'none', False, TOOLS, max_tokens=budget, thinking=thinking, parallel=True)
    remaining()
    if messages != original or event['HTTP_status'] != 200:
        raise ValueError('Second actual HTTP request failed or complete history mutated')
    usages.append(usage); events.append(event)
    bindings.append(response_binding(folder, 2, usage, messages, thinking))
    if usage['prompt_tokens'] < 8192 or any(u['prompt_tokens'] + budget > 32768 for u in usages):
        raise ValueError('Actual full tool history below 8K or context overflow')
    if thinking and not message.get('reasoning_content'):
        raise ValueError('Requested thinking not observed')
    check_final(message, finish, expected)
    owners = owner_receipts(pathlib.Path(args.service_report), args, session, bindings, budget, remaining)
    return {'mode': label, 'requested_thinking': thinking, 'actual_generation_thinking': True, 'thinking_toggle_qualified': False, 'passed': True, 'actual_HTTP_requests': 2, 'output_budget': budget,
        'same_response_two_actual_model_calls': True, 'unique_opaque_IDs': True,
        'tool_results_reversed_without_ID_replacement': True, 'file_nonce_sum_associations_correct': True,
        'wire': events, 'owner_receipts': owners,
        'file_SHA256': {name: sha(workspace/name) for name in PATHS}}


def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',required=True,type=float);p.add_argument('--timeout',default=7200,type=float);args=p.parse_args();deadline=args.deadline_unix;root=pathlib.Path(args.output_dir);root.mkdir(exist_ok=False)
 def remaining():
  left=deadline-time.time()
  if left<=120:raise TimeoutError('Original residency deadline; no reset/retry')
  return left
 q={'completed':False,'passed':False,'no_retries':True,'max_HTTP':4,'actual_HTTP_requests':0,'cases':[],'quality_scope':'Actual two-call and reverse original opaque-ID tool result association; full history/budgets retained'}
 try:
  remaining();ident=bind_identity(args.identity,args.manifest_sha,args.service_report);assert ident['actual_step']==args.expected_step;p0=load_wire(args.manifest_sha);rows=profiles(args.manifest_sha);profile=next(v for v in rows if v['label']=='HTTP');data=json.loads(pathlib.Path(profile['input_file']).read_bytes());q['source_profile_SHA256']=sha(pathlib.Path(profile['input_file']));q['identity_SHA256']=sha(pathlib.Path(args.identity));bound_report(pathlib.Path(args.service_report),args)
  OriginalWire=p0.WireClient
  class CountedWire(OriginalWire):
   def request(self,*args,**kwargs):
    q['actual_HTTP_requests']+=1
    return super().request(*args,**kwargs)
  p0.WireClient=CountedWire
  for thinking in [False,True]:q['cases'].append(run_mode(p0,args,root,data,thinking,remaining))
  q['passed']=len(q['cases'])==2 and all(v['passed']for v in q['cases'])and q['actual_HTTP_requests']==4
 except BaseException as e:q.update(error=repr(e))
 finally:q['completed']=True;(root/'result.json').write_text(json.dumps(q,indent=2)+chr(10))
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
