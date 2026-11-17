"""Distinct bounded completion: five immutable local pins, three exact new GETs."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from overlay_env import D,G,RUN,OVERLAY,environment
from job_guard import current_affinity
from windows_copy_identity import verified_copy


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,response,code,message,headers,newurl):return None


def need(ok,why):
    if not ok:raise ValueError(why)


def validate_response(status,headers,row,range_request=False):
    if range_request:
        need(status==206,'RANGE_MUST_RETURN206_NO200_FALLBACK')
        need(headers.get('Content-Range')=='bytes 3473408-8122770/8122771','EXACT_CONTENT_RANGE_REQUIRED')
        expected=4649363
    else:
        need(status==200 and headers.get('Content-Range') is None,'EXACT_COMPLETE_WHEEL200_REQUIRED')
        expected=row['bytes']
    need(headers.get('Content-Length')==str(expected),'EXACT_BODY_LENGTH_REQUIRED')
    need(headers.get('Content-Encoding') in (None,'identity'),'ENCODED_RANGE_OR_BODY_FORBIDDEN')
    return expected


def main(epoch):
    lock=json.loads((D/'WHEEL_LOCK.json').read_bytes());binding=json.loads((D/'ORIGINAL_BINDING.json').read_bytes())
    deadline=epoch+300;phase='JOB_GATE';pip_attempted=False;error=None;error_reason=None;passed=False;network_bytes=0
    attempts=[];copied={};whole={};upper=None;unpacked=0
    slots={name:dict(state='NOT_STARTED',planned_new_body_bytes=(4649363 if name=='matplotlib' else lock['wheels'][name]['bytes'])
                    if name in ('matplotlib','pyparsing','six') else 0) for name in lock['wheels']}
    def left():
        value=deadline-time.time()-10;need(value>0,'ORIGINAL_COMPLETION300_DEADLINE');return value
    def record_progress():
        value=dict(epoch=epoch,deadline=deadline,phase=phase,copied=copied,whole_wheels=whole,slots=slots,
                   new_GETs_planned=3,new_GETs_attempted=len(attempts),attempts=attempts,new_network_bytes=network_bytes,
                   new_network_cap=4774303,original_v1_attempted_GETs=5,combined_GET_denominator=8,pip_attempted=pip_attempted)
        temp=RUN/'completion-progress.tmp';temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temp.replace(RUN/'completion-progress.json')
    try:
        claim=json.loads((D/'CLAIM.json').read_bytes())
        need(claim['root_GO'] is True and claim['install_GO'] is True,'COMPLETION_SOURCE_ONLY_ROOT_GO_FALSE')
        gate=RUN/'job-gate.json'
        while not gate.exists():left();time.sleep(.02)
        owner=json.loads(gate.read_bytes())
        need(owner['deadline']==deadline and owner['CPU_count']==2 and owner['job_affinity_verified']
             and current_affinity()['process_mask']==owner['worker_affinity_mask'],'ACTUAL_NATIVE_JOB_GATE_REQUIRED')
        record_progress()
        for name,version in lock['existing_runtime'].items():need(md.version(name)==version,'EXISTING_RUNTIME_CHANGED')
        phase='COPY_ORIGINAL_FIVE_EXACT_PINS'
        for name,original in binding['original_files'].items():
            left();source=G/original['path'];target=RUN/'wheels'/lock['wheels'][name]['filename']
            verified=verified_copy(source,target,original,check=left)
            copied[name]=verified
            slots[name]['state']='PREFIX_COPIED' if name=='matplotlib' else 'WHOLE_VERIFIED'
            if name!='matplotlib':whole[name]=dict(bytes=verified['bytes'],SHA256=verified['SHA256'])
            record_progress()
        opener=urllib.request.build_opener(NoRedirect())
        phase='THREE_NEW_EXACT_GETS'
        for name in ('matplotlib','pyparsing','six'):
            left();row=lock['wheels'][name];ranged=name=='matplotlib';target=RUN/'wheels'/row['filename']
            expected=4649363 if ranged else row['bytes'];used=0
            headers={'Accept-Encoding':'identity','User-Agent':'Hygon-overlay-exact-completion/1.0'}
            if ranged:headers['Range']='bytes=3473408-8122770'
            request=urllib.request.Request(row['url'],headers=headers)
            attempt=dict(name=name,method='GET',range=headers.get('Range'),planned_body_bytes=expected,
                         received_body_bytes=0,state='ATTEMPTED')
            attempts.append(attempt);slots[name]['state']='GET_ATTEMPTED';record_progress()
            with opener.open(request,timeout=min(45,left())) as response:
                need(response.url==row['url'],'EXACT_OFFICIAL_URL_REQUIRED')
                validate_response(response.status,response.headers,row,ranged)
                if ranged:need(target.stat().st_size==3473408,'COPIED_PREFIX_SIZE_CHANGED')
                with target.open('ab' if ranged else 'xb') as output:
                    while True:
                        left();block=response.read(min(65536,expected-used+1))
                        if not block:break
                        used+=len(block);network_bytes+=len(block);attempt['received_body_bytes']=used
                        need(used<=expected and network_bytes<=4774303,'EXACT_NEW_NETWORK_BYTE_CAP')
                        need(output.write(block)==len(block),'SHORT_NEW_DOWNLOAD_WRITE')
                        record_progress()
                    output.flush();os.fsync(output.fileno())
            need(used==expected,'NEW_RESPONSE_TRUNCATED')
            raw=target.read_bytes();need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'],'FINAL_WHOLE_WHEEL_SHA')
            whole[name]=dict(bytes=len(raw),SHA256=row['SHA256']);slots[name]['state']='WHOLE_VERIFIED';attempt['state']='COMPLETE'
            record_progress()
        need(len(whole)==7 and len(attempts)==3 and network_bytes==4774303,'ALL7WHOLE_AND3NEW_GETS_REQUIRED_BEFORE_PIP')
        phase='ALL_SEVEN_WHOLE_SHA_AND_ZIP_BUDGET'
        import zipfile
        for name,row in lock['wheels'].items():
            path=RUN/'wheels'/row['filename'];raw=path.read_bytes()
            need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'],'PRE_PIP_WHOLE_WHEEL_SHA')
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    from pathlib import PurePosixPath
                    part=PurePosixPath(item.filename)
                    need(not part.is_absolute() and '..' not in part.parts and '\\' not in item.filename,'UNSAFE_WHEEL_MEMBER')
                    need((item.external_attr>>16)&0o170000!=0o120000,'WHEEL_SYMLINK')
                    unpacked+=item.file_size
        upper=2*unpacked+2*lock['wheel_download_bytes']+8*1024**2
        need(upper<=128*1024**2,'OWNED128MIB_PREINSTALL_BUDGET')
        (RUN/'wheel-layout.json').write_text(json.dumps(dict(whole=whole,uncompressed_bytes=unpacked,conservative_peak_bound_bytes=upper),indent=2)+'\n',encoding='utf8')
        need(left()>=30,'INSUFFICIENT_ORIGINAL_BUDGET_FOR_FIRST_PIP')
        phase='FIRST_PIP_INSTALL';pip_attempted=True;record_progress()
        command=[sys.executable,'-I','-B','-m','pip','--isolated','install','--no-index','--find-links',str(RUN/'wheels'),
                 '--only-binary=:all:','--no-deps','--require-hashes','--no-compile','--no-cache-dir',
                 '--target',str(OVERLAY),'--requirement',str(D/'requirements.lock')]
        env=environment();env['PIP_DISABLE_PIP_VERSION_CHECK']='1';env['PIP_CONFIG_FILE']=os.devnull
        with (RUN/'pip.stdout').open('xb') as stdout,(RUN/'pip.stderr').open('xb') as stderr:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,timeout=left())
        need(result.returncode==0,'FIRST_PIP_FAILED')
        phase='OVERLAY_FONT_SMOKE';record_progress()
        with (RUN/'smoke.stdout').open('xb') as stdout,(RUN/'smoke.stderr').open('xb') as stderr:
            result=subprocess.run([sys.executable,'-B','-X','utf8','-c',(D/'smoke_overlay.py').read_text()],
                 stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,timeout=left())
        need(result.returncode==0,'OVERLAY_FONT_SMOKE_FAILED');passed=True;phase='TERMINAL_PASS'
    except BaseException as failure:
        error=type(failure).__name__;error_reason=str(failure)[:240]
        if attempts and attempts[-1]['state']!='COMPLETE':attempts[-1]['state']='FAILED_OR_INCOMPLETE'
    finally:
        value=dict(passed=passed,error_type=error,error_reason=error_reason,phase=phase,epoch=epoch,deadline=deadline,copied=copied,whole_wheels=whole,
                   planned_wheels=7,completed_whole_wheels=len(whole),missing_whole_wheels=7-len(whole),slots=slots,
                   new_GETs_planned=3,new_GETs_attempted=len(attempts),attempts=attempts,new_network_bytes=network_bytes,
                   planned_new_network_bytes=4774303,original_v1_attempted_GETs=5,combined_GET_denominator=8,
                   combined_actual_attempts=5+len(attempts),original_RUN_immutable=True,pip_attempted=pip_attempted,
                   conservative_peak_bound_bytes=upper,shared_runtime_changed=False,global_environment_changed=False,
                   WDI_GETs=0,Torch_imports=0,GPU=0,replay_allowed=False)
        (RUN/'worker-result.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
        print(json.dumps({key:value[key] for key in ('passed','error_type','error_reason','phase','completed_whole_wheels','new_GETs_attempted','new_network_bytes','combined_GET_denominator','pip_attempted')}))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main(float(sys.argv[1])))
