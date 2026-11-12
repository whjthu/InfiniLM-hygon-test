from pathlib import Path
import hashlib,json,socket
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-original-model-v1');i=json.loads((W/'intent.json').read_bytes());o=json.loads((W/'outcome.json').read_bytes());assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST and i['epoch']==o['epoch']==EPOCH and i['deadline']==o['deadline']==DEADLINE==EPOCH+43200;assert o['passed'] and o['files']==PLAN['weights'] and o['weight_bytes']==8600277880 and o['GPU_cards']==0 and not o['model_loaded'] and o['wall_seconds']<43200;assert i['owner']==o['owner'];alive=None
if o['owner']['host']==socket.gethostname():
 try:s=Path('/proc',str(o['owner']['PID']),'stat').read_text().split(') ',1)[1].split();alive=int(s[19])==o['owner']['startticks']
 except FileNotFoundError:alive=False
 assert alive is False
files={n:dict(bytes=(W/n).stat().st_size,SHA256=hashlib.sha256((W/n).read_bytes()).hexdigest()) for n in ['intent.json','outcome.json']};print(json.dumps(dict(source_terminal_verified=True,original_outcome=o,original_artifacts=files,original_owner_alive=alive,reader_host=socket.gethostname(),weight_bodies_rehashed=False,GPU_cards=0)))
