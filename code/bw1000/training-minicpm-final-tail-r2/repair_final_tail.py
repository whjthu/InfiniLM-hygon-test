"""Complete only the terminal original copy's missing tail, inside its original lease."""
import base64,hashlib,json,shlex,subprocess,sys,threading,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
SENDER=r'''
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
assert EPOCH<time.time()<DEADLINE-30
signal.alarm(max(1,min(600,int(DEADLINE-time.time()-25))))
root=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-final-tail-r2');root.mkdir(mode=0o700)
identity={'claim_commit':CLAIM,'epoch':EPOCH,'deadline':DEADLINE,'host':socket.gethostname(),'PID':os.getpid(),'PGID':os.getpgrp(),'startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),'boot_id_SHA256':hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()}
with (root/'intent.json').open('x') as f:json.dump(identity,f)
source=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/MiniCPM-V-4.6/model.safetensors');before=source.stat()
assert before.st_size==TOTAL and not source.is_symlink()
with source.open('rb') as f:f.seek(OFFSET);tail=f.read(TOTAL-OFFSET+1)
after=source.stat()
assert len(tail)==TOTAL-OFFSET and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
tail_sha=hashlib.sha256(tail).hexdigest();header={'claim_commit':CLAIM,'offset':OFFSET,'total':TOTAL,'full_SHA256':FULL,'tail_bytes':len(tail),'tail_SHA256':tail_sha}
sys.stdout.buffer.write((json.dumps(header)+'\n').encode());sys.stdout.buffer.write(tail);sys.stdout.buffer.flush()
with (root/'receipt.json').open('x') as f:json.dump(header,f)
'''
RECEIVER=r'''
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
assert EPOCH<time.time()<DEADLINE-25
signal.alarm(max(1,min(600,int(DEADLINE-time.time()-20))))
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1')
assert hashlib.sha256((root/'resume-v1-failure.json').read_bytes()).hexdigest()=='4129c4c9cf275efcadf92b290b7e6d5fff3d9b03bddca64e7a5930197d3d15e3'
assert hashlib.sha256((root/'failure.json').read_bytes()).hexdigest()=='bc3442cd990d713dfd62951eb41cba50509c480e2f082ce80d647795cbf3d3c3'
repair=root/'tail-repair-v2';repair.mkdir(mode=0o700)
identity={'claim_commit':CLAIM,'epoch':EPOCH,'deadline':DEADLINE,'host':socket.gethostname(),'PID':os.getpid(),'PGID':os.getpgrp(),'startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),'boot_id_SHA256':hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()}
with (repair/'intent.json').open('x') as f:json.dump(identity,f)
try:
    header=json.loads(sys.stdin.buffer.readline(2048));assert header['claim_commit']==CLAIM and header['offset']==OFFSET and header['total']==TOTAL and header['full_SHA256']==FULL and header['tail_bytes']==TOTAL-OFFSET
    tail_data=sys.stdin.buffer.read(TOTAL-OFFSET+1);assert len(tail_data)==TOTAL-OFFSET and hashlib.sha256(tail_data).hexdigest()==header['tail_SHA256']
    with (repair/'tail.bin').open('xb') as f:f.write(tail_data);f.flush();os.fsync(f.fileno())
    model=root/'MiniCPM-V-4.6'
    for name,item in PINS.items():
        if name=='model.safetensors':continue
        raw=(model/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['SHA256']
    def journal(value):
        with (repair/'precommit.json').open('x') as f:json.dump(value,f)
    result=assemble(model/'model.safetensors.partial',repair/'tail.bin',model/'model.safetensors',TOTAL,FULL,header['tail_SHA256'],EXPECTED,journal)
    assert sorted(x.name for x in model.iterdir())==sorted(PINS)
    result.update(claim_commit=CLAIM,epoch=EPOCH,deadline=DEADLINE,files=PINS,original_failures_preserved=True,model_directory=str(model))
    with (repair/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
except BaseException as error:
    partial=root/'MiniCPM-V-4.6/model.safetensors.partial'
    with (repair/'failure.json').open('x') as f:json.dump({'claim_commit':CLAIM,'error_type':type(error).__name__,'partial_bytes':partial.stat().st_size if partial.exists() else None,'final_exists':(root/'MiniCPM-V-4.6/model.safetensors').exists()},f)
    raise
'''
def main():
    G=Path(sys.argv[1]);claim=sys.argv[2]
    def blob(name):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+name])
    task=json.loads(blob('coordination/tasks/training-minicpm-final-tail-r2.json'))
    source=blob(task['source']+'/repair_final_tail.py');assert source==Path(__file__).read_bytes() and hashlib.sha256(source).hexdigest()==task['source_SHA256']['repair_final_tail.py']
    helper=blob(task['source']+'/assemble_tail.py');assert hashlib.sha256(helper).hexdigest()==task['source_SHA256']['assemble_tail.py']
    epoch=task['original_epoch'];deadline=task['original_deadline'];assert epoch<time.time()<deadline-90
    pins=json.loads(blob(task['source']+'/MODEL_FILES.json'))['files'];weight=pins['model.safetensors']
    shared=f"CLAIM={claim!r}\nEPOCH={epoch!r}\nDEADLINE={deadline!r}\nOFFSET={task['retained_bytes']!r}\nTOTAL={weight['bytes']!r}\nFULL={weight['SHA256']!r}\nEXPECTED={task['retained_stat']!r}\nPINS={pins!r}\n"+helper.decode()+'\n'
    with (BASE/'launch-intent.json').open('x') as f:json.dump({'claim_commit':claim,'epoch':epoch,'deadline':deadline,'tail_bytes':weight['bytes']-task['retained_bytes']},f)
    ns={'__file__':str(Path(__file__).resolve())};prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
    exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
    dest_cmd=ns['ssh'].copy();dest_cmd[-1]='/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(shared+RECEIVER)
    config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
    source_cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26','python3 -B -u -c '+shlex.quote(shared+SENDER)]
    with (BASE/'source.stderr').open('xb') as se,(BASE/'receiver.stderr').open('xb') as de,(BASE/'receiver.stdout').open('xb') as out:
        dest=subprocess.Popen(dest_cmd,stdin=subprocess.PIPE,stdout=out,stderr=de);sender=subprocess.Popen(source_cmd,stdout=subprocess.PIPE,stderr=se)
        def stop_owned():
            for p in (sender,dest):
                if p.poll() is None:p.terminate()
        timer=threading.Timer(max(1,deadline-time.time()-15),stop_owned);timer.daemon=True;timer.start();copied=0;error=None
        try:
            while True:
                data=sender.stdout.read(65536)
                if not data:break
                dest.stdin.write(data);copied+=len(data)
            dest.stdin.close()
            sender.wait(timeout=max(1,deadline-time.time()-25));dest.wait(timeout=max(1,deadline-time.time()-20))
        except BaseException as e:
            error=type(e).__name__;stop_owned()
            for p in (sender,dest):
                try:p.wait(timeout=max(1,min(5,deadline-time.time()-5)))
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
        finally:timer.cancel()
    raw=(BASE/'receiver.stdout').read_bytes()
    result={'claim_commit':claim,'source_rc':sender.returncode,'destination_rc':dest.returncode,'local_error':error,'wire_bytes':copied,'epoch':epoch,'deadline':deadline,'local_wait_policy':'remaining original lease; no fixed20-second drain cutoff','stdout_SHA256':hashlib.sha256(raw).hexdigest()}
    if dest.returncode==0:result['receipt']=json.loads(raw)
    with (BASE/'outcome.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
