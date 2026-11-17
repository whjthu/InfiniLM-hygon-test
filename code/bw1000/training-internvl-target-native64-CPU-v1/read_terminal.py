"""Future independent original-step metadata audit, terminal accounting first."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scope import D,P,RUN,COLS,require,read,pin,save,verify
from slurm_accounting import is_terminal


def main(request,publication,epoch,step):
    require(step.startswith(P['parent']+'.') and step.rsplit('.',1)[1].isdigit(),'actual original CPU step required')
    # Before source/result reads, establish actual original Slurm terminal+resources.
    result=subprocess.run(['sacct','-n','-P','-j',step,'-o',COLS],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=12)
    require(result.returncode==0 and len(result.stdout)<=65536,'bounded independent accounting')
    rows=[parts for line in result.stdout.splitlines() if len(parts:=line.split('|'))==7 and parts[0]==step and parts[1]==P['label']]
    require(len(rows)==1 and is_terminal(rows[0]),'original actual terminal not established')
    row=rows[0];require(row[5]==P['node'] and set(row[6].split(','))=={'cpu=8','mem=12G','node=1'},'actual0GPU8CPU12G resource differs')
    mf=verify();intent=read(RUN/'intent.json');outcome=read(RUN/'outcome.json')
    for value in (intent,outcome):
        require(value['request_id']==request and value['publication']==publication and value['epoch']==epoch and value['deadline']==epoch+900,'original context differs')
    require(outcome['accounting']['row']==row and outcome['accounting']['terminal_visible'] is True,'producer/current accounting disagree')
    inside=read(RUN/'stage/outcome.json');before=read(RUN/'stage/ownership-pre.json');after=read(RUN/'stage/ownership-post.json')
    comparison=read(RUN/'stage/ownership-comparison.json')
    from ownership import validate_pair
    physical_post=False;post_error=None
    try:physical_post=validate_pair(before,after,P['parent'],P['node'],comparison['container_exit'])
    except (ValueError,KeyError,TypeError) as error:post_error=type(error).__name__
    worker=read(RUN/'worker-result.json') if (RUN/'worker-result.json').exists() else None
    rows_value=read(RUN/'native-caller-rows.json') if (RUN/'native-caller-rows.json').exists() else []
    full=bool(outcome['passed'] and row[2:4]==['COMPLETED','0:0'] and inside['passed'] and worker and worker['passed'])
    if full:
        require(len(rows_value)==64 and [item['ordinal'] for item in rows_value]==list(range(1,65)),'complete ordered64 rows')
        require(worker['completed_rows']==64 and worker['pre_model64_reference_exact'] and physical_post,'full CPU caller+post evidence differs')
        require(all(0<item['sequence_length']<=4096 and item['original_encoded_fields_exact']
                    and item['label_and_logit_mask_native_semantics_checked'] and item['complete_input_sequence_unchanged'] for item in rows_value),'original fullsequence/mask boundary differs')
        require(worker['actual_model_loaded'] is False and worker['image_embedding_forward'] is False
                and worker['training_forward_backward'] is False and worker['canonical_post_model64_equality_qualified'] is False,'CPU boundary overclaimed')
    pins={}
    for relative in ('intent.json','outcome.json','stage/outcome.json','stage/ownership-pre.json','stage/ownership-post.json',
                     'stage/ownership-comparison.json','worker-result.json','native-caller-rows.json','native-caller-providers.json',
                     'stage/worker.stdout','stage/container.stderr','stage.stdout','Slurm.stderr'):
        path=RUN/relative
        if path.exists():
            require(path.stat().st_size<=2*1024**2,'bounded original artifact')
            pins[relative]=pin(path)
    value=dict(request_id=request,publication=publication,epoch=epoch,deadline=epoch+900,actual_terminal=row,
               accounting_columns=COLS,source_MF=mf,independent_native_CPU64_verified=full,
               independent_post_verified=physical_post,post_error=post_error,planned_rows=64,completed_rows=len(rows_value),
               missing_rows=64-len(rows_value),original_artifact_pins=pins,GPU_GO=False,Model_GO=False,
               training_numerical_GO=False,canonical_post_model64_equality_qualified=False,replay_allowed=False)
    output=RUN/'independent-root-reader';output.mkdir();save(output/'result.json',value)
    print(json.dumps({key:value[key] for key in ('request_id','independent_native_CPU64_verified','independent_post_verified','planned_rows','completed_rows','missing_rows','GPU_GO')}))
    return 0 if full and physical_post else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--publication',required=True);parser.add_argument('--epoch',required=True,type=float);parser.add_argument('--step',required=True)
    a=parser.parse_args();raise SystemExit(main(a.request,a.publication,a.epoch,a.step))
