"""One fixed slot's container-only stdlib client argv; no tokenizer/Native."""
import hashlib
import json
import os
from pathlib import Path
import sys
import controller as C


def quality_args(slot,m,actual,epoch):
    # These paths intentionally address /refactor INSIDE the same established
    # container as CPU922. The keeper never runs this client on HOST paths.
    root=Path('/refactor');deploy=root/'incoming/GLM-W4-remaining-slots-r460/deployment-v1'
    provider=root/'incoming/GLM444-verified-ready-socket-CPU-v1/provider';folder=root/'results/GLM-W4-fewshot-quality-r444'/slot['run']
    return ['--base-url','http://127.0.0.1:43843/v1','--provider-root',str(provider),
        '--provider-source-sha',m['provider_source_SHA256'],'--audited-owner-sha',m['audited_owner_SHA256'],
        '--config',str(deploy/('config-slot%d.json'%slot['slot'])),'--config-sha',m['config_SHA256'],
        '--run',slot['run'],'--expected-step',actual,'--service-report',str(folder/'owner-1-0.json'),
        '--fixtures',str(root/'incoming/GLM-W4-fewshot-quality-r444/fixtures-r1'),
        '--encoding-receipt',str(root/'candidates/GLM-W4-fewshot-quality-r444-CPU-v1/encoding.json'),
        '--output-dir',str(root/'results'/(C.LABEL+'-'+slot['run']+'-client')),
        '--shard-index',str(slot['slot']),'--shards','8','--native-offset','0',
        '--deadline-unix',str(epoch+28200),'--gpu-start-epoch',str(epoch),'--timeout','5400']


def main():
    assert C.R==Path('/refactor'), 'Client dispatch is container-only; no implicit path replacement'
    assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    C.configure_slot(int(os.environ['GLM_R444_SLOT']));m=C.verify(os.environ['GLM_R444_DEPLOY_MF'])
    actual=os.environ['GLM_R444_ACTUAL_GPU'];assert actual.startswith(str(C.PARENT)+'.')
    epoch=float(os.environ['GLM_R425_GPUSTART']);args=quality_args(C.SLOT,m,actual,epoch)
    client=C.CLIENT_PACKAGE/'client-v1/quality_client.py';h=hashlib.sha256(client.read_bytes()).hexdigest()
    assert h==m['client_source_SHA256']
    identity=Path('/refactor/results')/(C.LABEL+'-'+C.RUN+'-client-identity.json')
    os.execv(sys.executable,[sys.executable,'-B','-u',str(C.D/'client_launch.py'),
        '--identity',str(identity),'--client',str(client),'--client-sha',h,*args])


if __name__=='__main__':main()
