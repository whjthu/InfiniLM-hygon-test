from pathlib import Path
import hashlib,io,json,signal,subprocess,sys,tempfile,time
D=Path(__file__).resolve().parent
source=(D/'receive.py').read_text()
for mode in ('good','truncated','wrong_sha','trailing'):
    with tempfile.TemporaryDirectory() as tmp:
        payloads={'config.json':b'{"x":1}', 'diffusion_pytorch_model.safetensors':b'payload'}
        raw=b''.join(payloads.values())
        pins={k:dict(bytes=len(v),SHA256=hashlib.sha256(v).hexdigest()) for k,v in payloads.items()}
        if mode=='truncated':raw=raw[:-1]
        if mode=='wrong_sha':raw=raw[:-1]+b'X'
        if mode=='trailing':raw+=b'!'
        root=Path(tmp)/'test'
        timer='import signal\nsignal.alarm=lambda n:None\nsignal.SIGALRM=signal.SIGINT\n' if not hasattr(signal,'alarm') else ''
        argv=[sys.executable,'-B','-c',timer+f"PLAN={dict(root=str(root),seconds=20,files=pins)!r}\n"
              f"DEADLINE={time.time()+18!r}\nCLAIM='test'\nREQUEST='test'\n"+source]
        r=subprocess.run(argv,input=raw,capture_output=True,timeout=10)
        assert (root/'outcome.json').is_file(),r.stderr.decode(errors='replace')
        out=json.loads((root/'outcome.json').read_bytes())
        assert (r.returncode==0)==out['passed']==(mode=='good')
        final=root/'hunyuanvideo-1.5/transformer/480p_t2v'
        assert final.is_dir()==(mode=='good')
        if mode=='good':assert all((final/k).read_bytes()==v for k,v in payloads.items())
print(json.dumps(dict(receiver_cases=4,passed=True,weight_payload_bytes=0)))
