from pathlib import Path
import re,ast,json,hashlib
root=Path(__file__).resolve().parent;src=root/'GLM-M16-Model-private-r8866';dst=root/'GLM-chunk128-Model-private-r8872';dst.mkdir(exist_ok=True)
for p in src.iterdir():
 if p.is_file():
  name=p.name.replace('r8866','r8872');b=p.read_bytes()
  if p.suffix in ('.py','.sh','.json'):b=b.replace(b'r8866',b'r8872')
  (dst/name).write_bytes(b)
(dst/'glm_chunk128_schedule_r8872.py').write_bytes((root/'glm_chunk128_schedule_r8872.py').read_bytes())
def replace(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)

# Preserve the original decode path by default, add only explicit resumption.
p=dst/'kimi_cohort_feedback_idle_r4915.py';s=p.read_text(encoding='utf-8')
s=replace(s,'clock=time.perf_counter, anchor=None):','clock=time.perf_counter, anchor=None, state=None, active_cohorts=None, max_turns=None, validate=None):')
s=replace(s,'    outstanding = [None] * len(cohorts)\n    next_step = [0] * len(cohorts)', '''    if state is None:state={}
    next_step=state.setdefault('next_step',[0]*len(cohorts))
    assert len(next_step)==len(cohorts) and all(type(x)is int and x>=0 for x in next_step)
    outstanding=[None]*len(cohorts)
    eligible=list(range(len(cohorts))) if active_cohorts is None else list(active_cohorts)
    assert len(set(eligible))==len(eligible) and all(0<=c<len(cohorts)for c in eligible)
    assert max_turns is None or(type(max_turns)is int and max_turns>0)
    turns=0''')
s=replace(s,'        assert packet["rows"] == rows and len(rows) == len(packet["ids"])','        assert packet["rows"] == rows and len(rows) == len(packet["ids"])\n        if validate is not None:validate(packet,rows)')
s=replace(s,'        for cohort, members in enumerate(cohorts):','        for cohort in eligible:\n            members=cohorts[cohort]')
s=replace(s,'            if outstanding[cohort] is not None:','            if outstanding[cohort] is not None:')
s=replace(s,'            next_step[cohort] += 1\n            launched = True','''            next_step[cohort] += 1
            turns+=1
            launched=True
            if max_turns is not None and turns>=max_turns:
                for pending,entry in enumerate(outstanding):
                    if entry is not None:
                        pending_step,pending_rows=entry
                        accept(feedback.request('take',case,cohort=pending,step=pending_step),pending_rows)
                        outstanding[pending]=None
                assert all(v is None for v in outstanding)
                return samples''')
ast.parse(s);(dst/'glm_chunk128_feedback_r8872.py').write_bytes(s.encode('utf-8'))
# The old shared basename remains byte-for-byte frozen. Only the new private
# module is imported by this experiment; staging must never overwrite it.
p.write_bytes((src/p.name).read_bytes())
p=dst/'glm_continuous_feedback_r8872.py';s=p.read_text(encoding='utf-8')
s=s.replace('from kimi_cohort_feedback_idle_r4915 import','from glm_chunk128_feedback_r8872 import')
ast.parse(s);p.write_bytes(s.encode('utf-8'))

p=dst/'glm_prefill_micro_probe_r8872.py';s=p.read_text(encoding='utf-8')
s=s.replace('import socket,secrets','import socket,secrets\nfrom glm_chunk128_schedule_r8872 import prompt_calls,pack_prompt,pack_decode')
s=s.replace("'kimi_cohort_feedback_idle_r4915.py']", "'glm_chunk128_feedback_r8872.py','glm_chunk128_schedule_r8872.py']")
s=replace(s,'prefill_micro=8):','prefill_micro=8,chunk_tokens=512):')
s=replace(s,'assert prefill_micro==8 and micro in(8,16)','assert prefill_micro==8 and micro==8 and chunk_tokens in(128,512)\n        if chunk_tokens==128:assert not qa and reference_kind==\'I512\' and len(prompts)==16 and limit==128')
# Define the existing checked callback before prompt scheduling so a completed
# cohort may use it between chunks. Do not mutate the outer prefill round.
a=s.index('                class ObservedFeedback(CheckedFeedback):');b=s.index('                safe_decode(',a)
callback=s[a:b];callback='\n'.join(line[8:] if line.startswith('        ') else line for line in callback.splitlines())+'\n'
callback=replace(callback,'            nonlocal step,saved_ids','            nonlocal saved_ids')
callback=replace(callback,'        pasts=[len(p)for p in prompts]\n','')
old='''            descriptions=[]; flat=[]; positions=[]
            for row in group:
                assert len(generated[row]) == step
                p = 0 if step == 0 else len(prompts[row])+step-1
                current = prompts[row] if step == 0 else [generated[row][-1]]
                descriptions.extend([row, p, len(current)])
                flat.extend(current if stage == 0 else [0]*len(current))
                positions.extend(range(p, p+len(current)))'''
callback=replace(callback,old,'            descriptions,flat,positions=pack_decode(prompts,generated,group,step,stage)')
s=s[:a]+s[b:]
s=replace(s,'        while not all(done):','        decode_state={}\n        pasts=[len(p)for p in prompts]\n'+callback+'        while not all(done):')
s=replace(s,"                assert all(len(g)==len(t)==1 for g,t in zip(generated,completions)),'prefill must produce exactly one token per request'","                assert all(len(g)==len(t)>=1 for g,t in zip(generated,completions)),'every row must complete512 before decode'\n                assert all(len(g)==(5 if chunk_tokens==128 and i<8 else 1) for i,g in enumerate(generated))")
s=replace(s,'                root_observer=[list(t)for t in completions]','                if chunk_tokens==512:root_observer=[list(t)for t in completions]')
s=replace(s,'                    forward=local_forward,anchor=start)','                    forward=local_forward,anchor=start,state=decode_state,validate=validate_decode)')
s=replace(s,'            for group in batches:','            calls=prompt_calls(batch,chunk_tokens) if chunk_tokens==128 and step==0 else [(g,0,512)for g in batches]\n            for group,prompt_offset,prompt_count in calls:\n                emit_generation=step>0 or chunk_tokens==512 or prompt_offset+prompt_count==512')
s=replace(s,'                descriptions=[]; flat=[]; positions=[]\n                for row in group:', '                descriptions=[]; flat=[]; positions=[]\n                for row in group:') # asserted original marker
s=replace(s,'                assert len(flat) <= 4096 and max(positions) < 1024\n                ids = torch.tensor(flat,', '                if step==0 and chunk_tokens==128:\n                    descriptions,flat,positions=pack_prompt(prompts,group,prompt_offset,prompt_count,stage)\n                assert len(flat) <= 4096 and max(positions) < 1024\n                ids = torch.tensor(flat,')
s=replace(s,'                if stage == PP-1:\n                    assert torch.isfinite(logits).all()', '                if stage == PP-1 and emit_generation:\n                    assert torch.isfinite(logits).all()')
s=replace(s,'                if not pipelined:\n                    packet = emitted[group].contiguous(); control(packet); feedback_calls += 1','                if (not pipelined or chunk_tokens==128) and emit_generation:\n                    packet = emitted[group].contiguous(); control(packet); feedback_calls += 1')
s=replace(s,'                    stamp = time.perf_counter()-start\n                    accept(generated, done, group, values, answer_eos, limit)',"                    if reference_kind=='I512' and step==0:\n                        expected=[historical_i512['generated_ids'][i][0]for i in group]\n                        if values!=expected:\n                            report['first_token_failure']={'phase':label,'rows':group,'actual':values,'expected':expected};save();raise CandidateRejected('chunk_first_token_gold_mismatch')\n                    stamp = time.perf_counter()-start\n                    accept(generated, done, group, values, answer_eos, limit)")
s=replace(s,"                events.append({'round': step, 'rows': group, 'flat_tokens':len(flat),", "                events.append({'prompt_offset':prompt_offset if step==0 else None,'prompt_count':prompt_count if step==0 else None,'round': step, 'rows': group, 'flat_tokens':len(flat),")
s=s.replace("events.append({'round': step,", "events.append({'requests':[descriptions[i:i+3]for i in range(0,len(descriptions),3)],'round': step,")
s=s.replace("events.append({'prompt_offset':", "events.append({'requests':[descriptions[i:i+3]for i in range(0,len(descriptions),3)],'prompt_offset':")
s=replace(s,'            if pipelined:\n                control(emitted);', '''                if chunk_tokens==128 and step==0 and group[0]==8:
                    # Complete the original publish/take cycle before the next
                    # prompt call; no simultaneous Runtime/PP call is introduced.
                    safe_decode(feedback=checked,case=case_name,is_emitter=participant==emitter,
                        ids=generated,pasts=pasts,times=completions,micro=micro,limit=limit,eos=answer_eos,
                        forward=local_forward,anchor=start,state=decode_state,active_cohorts=[0],max_turns=1,validate=validate_decode)
            if pipelined and chunk_tokens==512:
                control(emitted);''')
s=replace(s,"'prefill_micro':prefill_micro,'prefill_factor_arm'", "'prompt_chunk_tokens':chunk_tokens,'early_decode_outputs':sum(len(e['rows'])for e in events[:max(i for i,e in enumerate(events)if e['round']==0)+1]if e['round']>0),'prefill_micro':prefill_micro,'prefill_factor_arm'")
s=replace(s,"labels=['perf-I512-B32-M8-N-r0','perf-I512-B32-M16-N-r0']", "labels=['perf-I512-B16-M8-A-N-r0','perf-I512-B16-M8-C128-N-r0']")
s=replace(s,"            prompts=i512_inputs;micro=16 if '-M16-'in label else 8", "            prompts=i512_inputs[:16];micro=8")
s=replace(s,"generate(label,prompts,micro,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)","generate(label,prompts,micro,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8,128 if '-C128-'in label else 512)")
s=replace(s,"historical_i512=json.loads(frozen477.read_text());assert historical_i512['input_ids']==i512_inputs", "historical_i512=json.loads(frozen477.read_text());assert historical_i512['input_ids']==i512_inputs\n    historical_i512={**historical_i512,**{k:historical_i512[k][:16]for k in ['input_ids','generated_ids','stopped_at_EOS']}}")
s=replace(s,"    independent_references={}","    i512_inputs=i512_inputs[:16]\n    independent_references={}")
s=replace(s,'                    for row in group: completions[row].append(stamp)','                    for row in group:\n                        completions[row].append(stamp)\n                        if chunk_tokens==128:root_observer[row].append(stamp)')
s=replace(s,'        decode_state={}\n', '''        def validate_decode(packet,rows):
            if reference_kind=='I512':
                expected=[historical_i512['generated_ids'][r][len(generated[r])]for r in rows]
                if packet['ids']!=expected:
                    report['first_decode_failure']={'phase':label,'rows':rows,'output_indices':[len(generated[r])for r in rows],'actual':packet['ids'],'expected':expected};save()
                    raise CandidateRejected('decode_token_gold_mismatch')
        decode_state={}
''')
s=s.replace('Same-load B32M8->B32M16 scheduling-only full128','Same-load single16owner B16M8 ordinaryA thenchunk128C full128')
s=s.replace('ROWS32 TP8PP2/B32M8thenB32M16','ROWS32 TP8PP2/B16M8 ordinaryA thenchunk128C')
ast.parse(s);p.write_bytes(s.encode('utf-8'))
p=dst/'glm_crossrequest_control_r8872.py';s=p.read_text(encoding='utf-8')
s=replace(s,"sum(len(prompts[i])+e['round']for e in events for i in e['rows'])", "sum(q[1]+q[2]for e in events for q in e['requests'])")
ast.parse(s);p.write_bytes(s.encode('utf-8'))
print(json.dumps({'private':str(dst),'probe_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
