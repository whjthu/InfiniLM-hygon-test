import pathlib,subprocess,json
p=pathlib.Path(__file__).resolve().parent
cmd=['ssh','-T','-p','65032','-i','C:/Users/WangHaojie/.ssh/hygon-261012.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile=C:/Users/WangHaojie/Documents/ChatGPT/海光集群/known_hosts','aclsylqidf@zzeshell.scnet.cn','python3 -']
q=subprocess.run(cmd,input=(p/'read_binding_r121.py').read_text(),capture_output=True,text=True,encoding='utf8',timeout=100);assert q.returncode==0,q.stderr[-2000:]
data=json.loads(q.stdout);out=p/'actual-binding-r121.json';assert not out.exists();out.write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps({'saved':str(out),'files':len(data['files']),'receipts':len(data['receipts']),'bytes':out.stat().st_size}))
