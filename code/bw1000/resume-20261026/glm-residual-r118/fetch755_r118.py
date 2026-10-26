import pathlib,subprocess,json
folder=pathlib.Path(__file__).resolve().parent
code=(folder/'read755_safe_r118.py').read_text(encoding='utf8')
cmd=['ssh','-T','-p','65032','-i','C:/Users/WangHaojie/.ssh/hygon-261012.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile=C:/Users/WangHaojie/Documents/ChatGPT/海光集群/known_hosts','aclsylqidf@zzeshell.scnet.cn','python3 -']
p=subprocess.run(cmd,input=code,capture_output=True,text=True,encoding='utf8',timeout=90)
assert p.returncode==0,p.stderr[-1000:]
q=json.loads(p.stdout);out=folder/'allowner-scalar755-r118.json';assert not out.exists();out.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(saved=str(out),owners=len(q['owners']),calls=len(q['aligned_call_summary']),bytes=out.stat().st_size)))
