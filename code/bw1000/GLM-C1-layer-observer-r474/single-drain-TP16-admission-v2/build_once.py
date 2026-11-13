"""Verify the exact metadata and paired TCP path on the current I02 nodes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

root=Path(__file__).resolve().parents[1]
source=root/'src'
read=lambda path:json.loads(path.read_bytes())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
intent=read(root/'intent.json')
binding=read(source/'BINDING.json')
assert all(sha(source/name)==digest for name,digest in intent['source_files'].items())
node=socket.gethostname().split('.')[0]
rank=int(os.environ['SLURM_PROCID'])
nodes=binding['nodes']
assert nodes[rank]==node and len(nodes)==2
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
assert step.startswith('1103606.')
assert all(os.environ.get(name)=='-1' for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
model=Path(binding['model_path'])
assert all(sha(model/name)==digest for name,digest in binding['files'].items())
tokenizer=Path(binding['tokenizer']['path'])
assert tokenizer.stat().st_size==binding['tokenizer']['bytes'] and sha(tokenizer)==binding['tokenizer']['SHA256']
assert all(sha(Path(path))==digest for path,digest in binding['libraries'].items())
mapping=read(model/'model.safetensors.index.json')['weight_map']
assert all(Path(name).name==name for name in mapping.values())
shards={name:(model/name).stat().st_size for name in sorted(set(mapping.values()))}
assert shards and all(size>0 for size in shards.values())
addresses=[socket.gethostbyname(name) for name in nodes]
assert len(set(addresses))==2
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as reserve:
    reserve.bind((addresses[rank],43843))
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind((addresses[rank],43844))
        listener.listen(1)
        listener.settimeout(15)
        ready=root/f'listener-{rank}.json'
        with ready.open('x') as out:
            json.dump(dict(step=step,node=node,addresses=addresses),out)
        deadline=time.monotonic()+45
        other=1-rank
        listener.settimeout(45)
        message=lambda who: bytes.fromhex(intent['request_id'])+bytes([who])
        def expect(connection,expected):
            connection.settimeout(8)
            with connection.makefile('rb') as stream:
                assert stream.read(len(expected))==expected
        def receive():
            peer,remote=listener.accept()
            with peer:
                assert remote[0]==addresses[other]
                expect(peer,message(other))
                peer.sendall(message(rank))
        def send():
            while True:
                left=deadline-time.monotonic()
                if left<=0:
                    raise TimeoutError("TCP peer never accepted a connection")
                with socket.socket() as peer:
                    peer.settimeout(min(3,left))
                    peer.bind((addresses[rank],0))
                    try:
                        peer.connect((addresses[other],43844))
                    except (ConnectionRefusedError,TimeoutError):
                        time.sleep(.2)
                        continue
                    peer.sendall(message(rank))
                    expect(peer,message(other))
                    return
        if rank==0:
            send()
            receive()
        else:
            receive()
            send()
row=dict(passed=True,node=node,step=step,rank=rank,addresses=addresses,shard_count=len(shards),
         shard_bytes=sum(shards.values()),index_keys=len(mapping),shard_sizes_SHA256=hashlib.sha256(
             json.dumps(shards,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
         model_metadata_verified=True,tokenizer_verified=True,libraries_verified=len(binding['libraries']),
         TCP_both_directions=True,model_weights_read=False,GPU_cards=0)
with (root/f'rank-{rank}.json').open('x') as out:
    json.dump(row,out,indent=2)
