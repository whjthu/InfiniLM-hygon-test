"""Bounded transport attribution: <= 1 MiB public HTTP, 1 MiB SSH /dev/null."""
import datetime, hashlib, json, pathlib, shlex, subprocess, time, urllib.request
r = pathlib.Path(__file__).resolve().parent
out = r / 'DFlash2-transport-cause-r9010.json'
assert not out.exists()
rev = 'e77935fb4804e17eb55085bffd045eae1d779769'
url = f'https://huggingface.co/lightseekorg/kimi-k3-dflash2/resolve/{rev}/model.safetensors'
n = 1 << 20
d = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), revision=rev,
         public_source=url, max_HTTP_body_bytes=n, local_payload_files=False,
         source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
start = time.monotonic()
try:
    req = urllib.request.Request(url, headers={'Range': f'bytes=158446986-{158446986+n-1}'})
    with urllib.request.urlopen(req, timeout=35) as res:
        headers_wall = time.monotonic()-start
        first = res.read(1)
        first_wall = time.monotonic()-start
        remaining = n-len(first)
        count = len(first)
        h = hashlib.sha256(first)
        while remaining:
            b = res.read(min(65536, remaining))
            if not b: break
            h.update(b); count += len(b); remaining -= len(b)
        total = time.monotonic()-start
        d['HTTP'] = dict(status=res.status, content_range=res.headers.get('Content-Range'),
                         content_length=res.headers.get('Content-Length'),
                         accept_ranges=res.headers.get('Accept-Ranges'),
                         headers_wall_s=headers_wall, first_byte_wall_s=first_wall,
                         body_read_wall_s=total-headers_wall, total_wall_s=total,
                         bytes=count, body_read_bytes_per_s=count/max(total-headers_wall, 1e-9),
                         body_SHA256=h.hexdigest())
except BaseException as e:
    d['HTTP'] = dict(error_type=type(e).__name__, total_wall_s=time.monotonic()-start)
remote = '''import sys,time,json
print(json.dumps({'ready':True}),flush=True)
t=time.monotonic();n=0
with open('/dev/null','wb') as sink:
 while True:
  b=sys.stdin.buffer.read(65536)
  if not b:break
  sink.write(b);n+=len(b)
print(json.dumps({'bytes':n,'receive_wall_s':time.monotonic()-t}),flush=True)
'''
args = ['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),
        '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
        '-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts'),
        'aclsylqidf@zzeshell.scnet.cn','python3 -c '+shlex.quote(remote)]
p = None
start = time.monotonic()
try:
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ready = json.loads(p.stdout.readline())
    assert ready['ready']
    admission = time.monotonic()-start
    transfer = time.monotonic()
    p.stdin.write(bytes(n));p.stdin.close();p.stdin=None
    stdout, _ = p.communicate(timeout=30)
    elapsed = time.monotonic()-transfer
    result = json.loads(stdout.decode().strip())
    d['SSH'] = dict(exit_code=p.returncode, admission_wall_s=admission,
                    transfer_and_ACK_wall_s=elapsed, bytes_per_s=n/elapsed,
                    remote=result, destination='/dev/null', zero_bytes_sent=n)
except BaseException as e:
    d['SSH'] = dict(error_type=type(e).__name__, total_wall_s=time.monotonic()-start)
    if p is not None and p.poll() is None: p.kill();p.communicate()
old = r/'stream_DFlash2_checkpoint_r9004.py'
d['old_stream_source_SHA256'] = hashlib.sha256(old.read_bytes()).hexdigest()
d['old_stream_observations'] = [
 'HTTP read(8MiB) and SSH stdin.write(8MiB) are sequential with one combined timer; old receipt cannot attribute the bottleneck.',
 'urllib.request.urlopen uses its default proxy selection; this probe does not read or expose proxy/credential environment values.',
 'Old loop accepts a short HTTP EOF until the final size assertion; partial file remains intentionally.',
 'Default offset-zero full GET can be affected differently from a bounded Range GET; this 1MiB measurement is not proof of sustained 7.39GB throughput.'
]
out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
print(json.dumps(d))
