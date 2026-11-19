"""At most three RO queries of the same actual step; never a Slurm resubmission."""
import hashlib,re,subprocess,time
from common import need,save
from control_evidence import persist_bytes

def observe_actual_step(root,context,row,step,attempts):
    need(re.fullmatch(re.escape(row['parent'])+r'\.\d+',step),'EXACT_OBSERVED_COMPUTE_STEP')
    began=time.monotonic();local_end=began+min(15,context['epoch']+180-time.time())
    for ordinal in range(1,4):
        remaining=min(local_end-time.monotonic(),context['epoch']+180-time.time())
        need(remaining>0,'STEP_VISIBILITY_ORIGINAL_ADMISSION_DEADLINE')
        record=dict(context,ordinal=ordinal,requested_step=step,query_timeout_seconds=min(5,remaining),query_returncode=None,matched_records=None,stdout_lines=None,error_type=None)
        raw=err=b'';mark=time.monotonic();failure=None
        try:
            q=subprocess.run(['scontrol','show','step',step,'--oneliner'],stdin=subprocess.DEVNULL,capture_output=True,timeout=record['query_timeout_seconds'])
            raw,err=q.stdout,q.stderr;record['query_returncode']=q.returncode
        except BaseException as e:
            raw=getattr(e,'stdout',None) or getattr(e,'output',None) or b'';err=getattr(e,'stderr',None) or b''
            record['error_type']=type(e).__name__;failure=e
        record['query_wall_seconds']=time.monotonic()-mark
        # Includes zero bytes, nonzero command exits, oversized and malformed output.
        stem='actual-step-query-%02d'%ordinal
        record['stdout']=persist_bytes(root,stem+'.stdout.private',raw,262144,context)
        record['stderr']=persist_bytes(root,stem+'.stderr.private',err,262144,context)
        fields=None
        try:
            need(failure is None and record['query_returncode']==0,'ACTUAL_STEP_QUERY_FAILED')
            need(len(raw)+len(err)<=262144,'ACTUAL_STEP_QUERY_OUTPUT_BOUND')
            lines=raw.decode('utf8').strip().splitlines();record['stdout_lines']=len(lines);matches=[]
            for line in lines:
                tokens=[x.split('=',1) for x in line.split() if '=' in x]
                ids=[v for k,v in tokens if k=='StepId']
                need(len(ids)<=1,'MULTIPLE_STEP_IDS_ON_ONE_LINE')
                if ids==[step]:
                    need(len(tokens)==len({k for k,v in tokens}),'DUPLICATE_ACTUAL_STEP_FIELDS')
                    matches.append(dict(tokens))
            record['matched_records']=len(matches)
            need(len(matches)<=1,'DUPLICATE_EXACT_ACTUAL_STEP_RECORD')
            if matches:fields=matches[0];record['selected_fields']=fields
        except BaseException as e:
            failure=e;record['error_type']=type(e).__name__;record['error_reason_SHA256']=hashlib.sha256(str(e).encode()).hexdigest()
        finally:
            record['total_elapsed_seconds']=time.monotonic()-began
            attempts.append(record);save(root/(stem+'.json'),record)
        if failure is not None:raise failure
        need(time.monotonic()<=local_end and time.time()<context['epoch']+180,'STEP_VISIBILITY_ORIGINAL_ADMISSION_DEADLINE')
        if fields is not None:return fields
        if ordinal<3:
            delay=min(.2,local_end-time.monotonic(),context['epoch']+180-time.time())
            need(delay>0,'STEP_VISIBILITY_ORIGINAL_ADMISSION_DEADLINE');time.sleep(delay)
    raise ValueError('EXACT_ACTUAL_STEP_NOT_VISIBLE_AFTER_THREE_QUERIES')
