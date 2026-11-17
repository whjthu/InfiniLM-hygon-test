"""Bounded private exception metadata only; caller never publishes message/trace."""
import json
PHASES={'admission','import','initial','checkpoint20','checkpoint25'}


def clipped_utf8(value,cap=4096):
    return str(value).encode('utf-8',errors='replace')[:cap].decode('utf-8',errors='ignore')

def failure_record(error_type,phase,active_state_file,message,trace):
    if phase not in PHASES or not isinstance(error_type,str) or len(error_type.encode())>128:raise ValueError('Failure type/phase outside contract')
    if not isinstance(message,str) or not isinstance(trace,str) or len(message.encode())>4096 or len(trace.encode())>4096:raise ValueError('Failure diagnostic per-field bound exceeded')
    value=dict(error_type=error_type,phase=phase,active_state_file=active_state_file,message=message,trace=trace,
        private_only=True,raw_tensor_or_state_payload_saved=False)
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    if len(raw)>16384:raise ValueError('Whole private diagnostic bound exceeded')
    return raw
