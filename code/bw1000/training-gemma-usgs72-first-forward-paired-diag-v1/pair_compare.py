"""Strict hash/metadata comparison of two new independent checkpoint20 resumes."""
import json
import math
from pathlib import Path


def read(path):
    raw=Path(path).read_bytes()
    if not 0<len(raw)<=8*1024**2:raise ValueError('bounded pair metadata')
    return json.loads(raw)


def compare_pair(root):
    root=Path(root);differences=[];complete=[];loss_gaps=[];RNG_pairs=0;used_processes=set()
    def compare(field,left,right,rank,**where):
        if left!=right:differences.append(dict(field=field,rank=rank,**where))
    for rank in range(8):
        try:
            outputs=[];states=[];taps=[];owners=[]
            for arm in ('untraced','traced'):
                folder=root/arm/'train/resume-witness'/('rank-%02d'%rank)
                result=read(folder/'result.json');restored=read(folder/'restored-before-RNG.json')
                owner=read(folder/'owner.json');tap=read(root/arm/'common-rng'/('rank-%02d'%rank)/'result.json')
                if not(result['rank']==rank and result['world_size']==8 and result['completed_global_step']==25
                    and len(result['inputs'])==len(result['outputs'])==20
                    and [s['global_step'] for s in result['steps']]==[21,22,23,24,25]
                    and result['native_resume_window_complete'] is True and result['saved_checkpoints']==[25]
                    and restored['restored_parameter_optimizer_scheduler_equal'] is True
                    and len(result['RNG_loads'])==1 and result['RNG_loads'][0]['saved_equal'] is True
                    and result['RNG_loads'][0]['global_step']==20
                    and result['RNG_loads'][0].get('actual')
                    and tap['complete'] is True and len(tap['rows'])==8):raise ValueError('incomplete new5 rank')
                expected_tap=[(i,event,20,rank) for i in range(1,5) for event in ('entry','exit')]
                if [(v['ordinal'],v['event'],v['global_step'],v['rank']) for v in tap['rows']]!=expected_tap:
                    raise ValueError('first4 RNG point denominator invalid')
                if not any(s['state']['adapter']!=restored['actual']['adapter'] for s in result['steps']):
                    raise ValueError('no parameter movement in native5')
                for row in result['steps']:
                    if set(row['state'])!={'adapter','optimizer','scheduler'} or not all(v['finite'] is True for v in row['state'].values()):
                        raise ValueError('nonfinite/incomplete native state')
                if read(folder/'native-data-skip.json')!=dict(rank=rank,before=100,skipped=80,after=20):
                    raise ValueError('native cursor differs')
                outputs.append(result);states.append(restored['actual']);taps.append(tap);owners.append(owner)
            keys={(o['PID'],o['startticks']) for o in owners}
            if len(keys)!=2 or keys.intersection(used_processes) or any(
                o['rank']!=rank or o['local_rank']!=rank or o['PID']<=0 or o['startticks']<=0 for o in owners):
                raise ValueError('independent rank process identity missing')
            used_processes.update(keys)
            compare('restored20_state',states[0],states[1],rank)
            compare('loaded20_RNG',outputs[0]['RNG_loads'],outputs[1]['RNG_loads'],rank)
            for index,(a,b) in enumerate(zip(outputs[0]['inputs'],outputs[1]['inputs']),1):
                if a['ordinal']!=index or b['ordinal']!=index or a['global_step']!=20+(index-1)//4 or b['global_step']!=a['global_step']:
                    raise ValueError('actual native order/cursor invalid')
                if a['full_input']['finite'] is not True or b['full_input']['finite'] is not True:
                    raise ValueError('nonfinite full inputs')
                compare('full_input',a['full_input'],b['full_input'],rank,ordinal=index)
            for index,(a,b) in enumerate(zip(outputs[0]['outputs'],outputs[1]['outputs']),1):
                if a['ordinal']!=index or b['ordinal']!=index or a['loss']['finite'] is not True or b['loss']['finite'] is not True:
                    raise ValueError('loss identity/finite denominator invalid')
                gap=abs(a['loss_scalar']-b['loss_scalar'])
                if not math.isfinite(gap):raise ValueError('nonfinite loss gap')
                loss_gaps.append(gap);compare('loss',(a['loss'],a['loss_scalar']),(b['loss'],b['loss_scalar']),rank,ordinal=index)
            for a,b in zip(outputs[0]['steps'],outputs[1]['steps']):
                compare('updated_state',a['state'],b['state'],rank,global_step=a['global_step'])
            for a,b in zip(taps[0]['rows'],taps[1]['rows']):
                if (a['ordinal'],a['event'],a['global_step'])!=(b['ordinal'],b['event'],b['global_step']):
                    raise ValueError('common first4 boundary mismatch')
                compare('first4_RNG',a['RNG'],b['RNG'],rank,ordinal=a['ordinal'],event=a['event']);RNG_pairs+=1
            rng=[]
            for arm in ('untraced','traced'):
                rng.append(read(root/arm/'train/resume-witness'/('rank-%02d'%rank)/'checkpoint-25-rng.json')['RNG'])
            compare('final25_RNG',rng[0],rng[1],rank)
            complete.append(rank)
        except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
            differences.append(dict(field='missing_or_invalid_rank',rank=rank,error_type=type(error).__name__))
    identity_fields={'restored20_state','loaded20_RNG','full_input','first4_RNG','final25_RNG','missing_or_invalid_rank'}
    return dict(planned_ranks=8,completed_ranks=complete,planned_input_loss_pairs=160,
        compared_input_loss_pairs=len(complete)*20,planned_state_pairs=40,compared_state_pairs=len(complete)*5,
        planned_first4_RNG_pairs=64,compared_first4_RNG_pairs=RNG_pairs,
        same_independent_input_rng_qualified=len(complete)==8 and not any(d['field'] in identity_fields for d in differences),
        observed_noninterference_exact=len(complete)==8 and not differences,differences=differences,
        maximum_absolute_loss_gap=max(loss_gaps,default=None),original860_recovery_requalified=False,
        Model_GO=False,formal17=False,effect_qualified=False)
