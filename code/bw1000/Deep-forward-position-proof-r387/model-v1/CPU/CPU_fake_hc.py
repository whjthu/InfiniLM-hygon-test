"""Only CPU test leaves. Real C++ mode/counter contract is separately executed in CPU724."""
from types import SimpleNamespace as NS
def fake_lib(reset,scenario):
    state={'mode':0,'set_modes':[],'positions_mode':0,'positions_set_modes':[],'positions_counts':[[1 if scenario=='positions-ready-failure'and d==0 else 0,0,0,0,0]for d in range(8)],'counts':[[86,85 if scenario=='HC-ready-failure'else 86,0,0,0,0,0]for _ in range(8)]}
    def setmode(mode):state['mode']=mode;state['set_modes'].append(mode);return 0
    def getmode():return state['mode']
    def read(out,entries):
        assert entries==56
        for d in range(8):
            for k in range(7):out[d*7+k]=state['counts'][d][k]
        return 0
    def set_positions(mode):state['positions_mode']=mode;state['positions_set_modes'].append(mode);return 0
    def get_positions():return state['positions_mode']
    def read_positions(d,out,n):
        assert n==5
        for k in range(5):out[k]=state['positions_counts'][d][k]
        return 0
    def advance(tokens):
        for row in state['positions_counts']:
            if state['positions_mode']:
                row[0]+=43
                if scenario!='positions-hit-failure':row[1]+=62;row[2]+=496*tokens
            else:row[4]+=62
        for row in state['counts']:
            row[2]+=86
            if tokens==1:
                row[3]+=86
                if state['mode']and scenario!='HC-hit-failure':row[4]+=86
            else:row[5]+=86
    return NS(modern_native_engine_reset=reset,deep_hc_model_epilogue_set_r366=setmode,deep_hc_model_epilogue_get_r366=getmode,deep_hc_model_epilogue_counters_r366=read,deep_position_proof_set_r387=set_positions,deep_position_proof_get_r387=get_positions,deep_position_proof_stats_r387=read_positions,advance=advance,state=state)

def attach_client_hc(row,mode):
    tokens=[r['tokens']for r in row['Native_continuation']if r['kind']=='prefill']+[1]*(row['actual_generated_tokens']-1)
    before=[[86,86,0,0,0,0,0]for _ in range(8)];after=[r.copy()for r in before];coverage=[]
    for t in tokens:
        delta=[0,0,86,86 if t==1 else 0,86 if t==1 and mode else 0,86 if t!=1 else 0,0]
        for d in range(8):after[d]=[a+b for a,b in zip(after[d],delta)]
        coverage.append(dict(input_tokens=t,mode=mode,per_owner_delta=[delta.copy()for _ in range(8)],candidate_completed_branch=t==1 and mode==1,not_hardware_GPU_hit_counter=True))
    row.update(HC_mode=mode,HC_counters_before_request=before,HC_counters_after_request=after,HC_forward_coverage=coverage,HC_counter_observation_s=0.)

def attach_client_positions(row,mode):
    tokens=[('prefill',r['tokens'])for r in row['Native_continuation']if r['kind']=='prefill']+[('decode',1)]*(row['actual_generated_tokens']-1)
    before=[[0]*5 for _ in range(8)];after=[r.copy()for r in before];coverage=[]
    for kind,t in tokens:
        delta=[43,62,496*t,0,0]if mode else[0,0,0,0,62]
        for d in range(8):after[d]=[a+b for a,b in zip(after[d],delta)]
        coverage.append(dict(kind=kind,input_tokens=t,mode=mode,per_owner_delta=[delta.copy()for _ in range(8)],actual_Runtime_owner_HOST_branch_counts=True))
    groups={kind:{'forwards':sum(v['kind']==kind for v in coverage),'per_owner_sum':[[sum(v['per_owner_delta'][d][k]for v in coverage if v['kind']==kind)for k in range(5)]for d in range(8)]}for kind in ['prefill','decode']}
    row.update(positions_mode=mode,positions_counters_before_request=before,positions_counters_after_request=after,positions_forward_coverage=coverage,positions_counts_by_kind=groups,positions_counter_observation_s=0.)
