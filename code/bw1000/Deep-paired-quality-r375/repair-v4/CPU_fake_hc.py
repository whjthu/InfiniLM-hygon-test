"""Only CPU test leaves. Real C++ mode/counter contract is separately executed in CPU724."""
from types import SimpleNamespace as NS
def fake_lib(reset,scenario):
    state={'mode':0,'set_modes':[],'counts':[[86,85 if scenario=='HC-ready-failure'else 86,0,0,0,0,0]for _ in range(8)]}
    def setmode(mode):state['mode']=mode;state['set_modes'].append(mode);return 0
    def getmode():return state['mode']
    def read(out,entries):
        assert entries==56
        for d in range(8):
            for k in range(7):out[d*7+k]=state['counts'][d][k]
        return 0
    def advance(tokens):
        for row in state['counts']:
            row[2]+=86
            if tokens==1:
                row[3]+=86
                if state['mode']and scenario!='HC-hit-failure':row[4]+=86
            else:row[5]+=86
    return NS(modern_native_engine_reset=reset,deep_hc_model_epilogue_set_r366=setmode,deep_hc_model_epilogue_get_r366=getmode,deep_hc_model_epilogue_counters_r366=read,advance=advance,state=state)

def attach_client_hc(row,mode):
    tokens=[r['tokens']for r in row['Native_continuation']if r['kind']=='prefill']+[1]*(row['actual_generated_tokens']-1)
    before=[[86,86,0,0,0,0,0]for _ in range(8)];after=[r.copy()for r in before];coverage=[]
    for t in tokens:
        delta=[0,0,86,86 if t==1 else 0,86 if t==1 and mode else 0,86 if t!=1 else 0,0]
        for d in range(8):after[d]=[a+b for a,b in zip(after[d],delta)]
        coverage.append(dict(input_tokens=t,mode=mode,per_owner_delta=[delta.copy()for _ in range(8)],candidate_completed_branch=t==1 and mode==1,not_hardware_GPU_hit_counter=True))
    row.update(HC_mode=mode,HC_counters_before_request=before,HC_counters_after_request=after,HC_forward_coverage=coverage,HC_counter_observation_s=0.)
