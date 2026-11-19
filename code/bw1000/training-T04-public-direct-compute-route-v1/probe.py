"""One exact public-byte connectivity check from the assigned T04 compute node."""

import errno
import hashlib
import http.client
import json
import os
import signal
import socket
import time
from urllib.parse import urlsplit


def check(condition, code):
    if not condition:
        raise ValueError(code)


def get_first_byte(host, path, timeout):
    connection = http.client.HTTPSConnection(host, 443, timeout=timeout)
    try:
        connection.request("GET", path, headers={"Range":"bytes=0-0","Accept-Encoding":"identity"})
        response = connection.getresponse()
        return (response.status, response.getheader("Location"), response.getheader("Content-Range"),
                response.getheader("Content-Length"), response.read(1) if response.status == 206 else b"")
    finally:
        connection.close()


def main():
    check(socket.gethostname().split('.')[0] == "a14r3n14" and os.environ.get("SLURM_JOB_ID") == "1087656"
          and os.environ.get("SLURM_STEP_ID","").isdecimal(), "ASSIGNED_T04_SLURM_NODE")
    step = "1087656."+os.environ["SLURM_STEP_ID"]
    started = time.monotonic()
    first_path="/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/resolve/e137910552d616eb0f7305147ca054311a90c542/model-00001-of-00004.safetensors"
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,20)
    result = dict(step=step,node="a14r3n14",GPU=0,remote_persisted_tensor_bytes=0,
                  body_bytes=0, direct_range_ready=False, initial_host="huggingface.co")
    phase="official_HF"
    try:
        status,location,content_range,length,data=get_first_byte("huggingface.co",first_path,8)
        result["first_status"]=status
        if status in (301,302,303,307,308):
            target=urlsplit(location)
            check(target.scheme=="https" and target.hostname=="us.aws.cdn.hf.co" and
                  target.port in (None,443) and not target.username and not target.password,
                  "PINNED_OFFICIAL_CDN")
            phase="official_CDN"
            result["cdn_host"]=target.hostname
            path=target.path+('?' + target.query if target.query else '')
            status,_,content_range,length,data=get_first_byte(target.hostname,path,8)
        result.update(final_status=status,body_bytes=len(data))
        check(status==206 and content_range=="bytes 0-0/4995008464" and length=="1" and len(data)==1,
              "EXACT_PUBLIC_BYTE_RANGE")
        result.update(direct_range_ready=True, sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        detail=getattr(error,"errno",None)
        result.update(error_phase=phase,error_type=type(error).__name__,errno=detail,
                      error_is_network_unreachable=detail in (errno.ENETUNREACH,errno.EHOSTUNREACH))
    result["wall_seconds"]=time.monotonic()-started
    print(json.dumps(result,allow_nan=False,sort_keys=True))


if __name__=="__main__":
    main()
