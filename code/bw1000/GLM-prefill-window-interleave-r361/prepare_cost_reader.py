"""Reuse355's real-clock cost reader; keep complete token timelines primary."""
from pathlib import Path
D=Path(__file__).resolve().parent;BASE=D.parent/'GLM-two-cohort-Model-r355'
s=(BASE/'audit_cost.py').read_text().replace('actual r355','actual r361')
s=s.replace("TTFT_s=visible[0]-admission,user_decode_tok_s=", "TTFT_s=visible[0]-admission,first_to_last_s=visible[-1]-visible[0],user_decode_tok_s=")
s=s.replace('includes_serial_other_row_prefill_wait=True','includes_other_row_prefill_wait=True,timeline_not_trimmed=True')
s=s.replace("max32_decode_schedule_wall_s=dec,", "max32_remaining_decode_schedule_wall_s=dec,remaining_decode_is_not_whole_comparable_window=True,")
s=s.replace("scope='One static two-request six-pair Model pilot; no formal benchmark/onlineSLO'", "scope='361 static two-request six-pair Model pilot; fixeddecode-deferON, onlyprefillinterleave changes; no formal benchmark/onlineSLO'")
needle="    Path(a.output).write_text(json.dumps(q,indent=2)+'\\n',encoding='utf8')"
assert needle in s
extra='''    primary={}
    for row in [0,1]:
        fields=['TTFT_s','first_to_last_s','user_decode_tok_s','ITL_P95_s','ITL_P99_s','ITL_max_s']
        primary[str(row)]={field:{'OFF':[q['emitter'][i]['requests'][row][field]for i in [2,5]],'ON':[q['emitter'][i]['requests'][row][field]for i in [3,4]],'OFF_mean':statistics.mean(q['emitter'][i]['requests'][row][field]for i in [2,5]),'ON_mean':statistics.mean(q['emitter'][i]['requests'][row][field]for i in [3,4])}for field in fields}
    q.update(primary_full_request_metrics=primary,decode_defer_fixed_ON=True,changed_factor='prefill_interleave_at_complete_long_W4_boundaries',remaining_decode_window_is_not_total_decode_gain=True)
'''
s=s.replace(needle,extra+needle);compile(s,'audit_cost_r361.py','exec');(D/'audit_cost_r361.py').write_bytes(s.encode())
print('361 whole-timeline reader prepared; no Runtime/GPU execution')
