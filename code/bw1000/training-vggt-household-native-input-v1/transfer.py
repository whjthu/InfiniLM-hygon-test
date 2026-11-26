"""Relay only the pinned official TUM members on the existing authenticated project SSH route."""

import argparse
import io
import json
from pathlib import Path
import queue
import runpy
import shlex
import subprocess
import tarfile
import threading
import time
from uuid import uuid4

import carrier
from native_entry import find


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPTURE = REPO/'local-only/vggt-r47-3493-household-eval-v1/official-frozen20-media-v2-once/unqualified-originals'
SSH_HELPER = REPO/'local-only/run-minicpm-sdpa-target-asset-preflight-original.py'
SSH_HELPER_SHA = '530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd'


def make_tar():
    files = carrier.input_spec((HERE/'INPUT_FILES.json').read_bytes())
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w', format=tarfile.USTAR_FORMAT) as stream:
        for name,expected in sorted(files.items()):
            value = find(CAPTURE/name, expected['SHA256'], expected['bytes'])
            item = tarfile.TarInfo(name)
            item.mode = 0o600
            item.size = len(value)
            stream.addfile(item,io.BytesIO(value))
    raw = archive.getvalue()
    carrier.need(len(raw) <= carrier.TAR_CAP and len(raw)%512 == 0, 'approved_tar_size')
    return raw


def communicate(raw, packet, destination):
    carrier.need(carrier.pin(SSH_HELPER.read_bytes())['SHA256'] == SSH_HELPER_SHA,'exact_existing_SSH_helper')
    ssh = runpy.run_path(str(SSH_HELPER))['ssh'](REPO.parent)
    argv = [carrier.MANAGED,'-I','-S','-B','-u',str(carrier.RUN/'carrier.py'),'transfer',packet['request'],
            packet['nonce'],packet['installed_SHA256'],packet['tar_SHA256'],str(packet['bytes'])]
    carrier.write(destination/'intent.json',carrier.encode(packet))
    process = subprocess.Popen([*ssh,shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    events = queue.Queue()

    def first_line():
        try:
            events.put((True,process.stdout.readline(8193)))
        except OSError as error:
            events.put((False,type(error).__name__))

    reader = threading.Thread(target=first_line,daemon=True)
    reader.start()
    result = {'request':packet['request'],'authority_verified':False,'payload_send_attempted':False,
              'outcome':'UNOBSERVED','response':None}
    out = err = b''
    try:
        ok,line = events.get(timeout=22)
        reader.join(timeout=1)
        carrier.need(ok and not reader.is_alive() and 0 < len(line) <= 8192, 'bounded_fresh_authority')
        authority = json.loads(line)
        carrier.need(authority == dict(packet,event='BOUND'), 'same_session_gate_identity')
        result['authority_verified'] = True
        carrier.write(destination/'send.json',carrier.encode(authority))
        result['payload_send_attempted'] = True
        out,err = process.communicate(input=raw,timeout=125)
        result['rc'] = process.returncode
        carrier.need(process.returncode == 0 and 0 < len(out) <= 8192, 'exact_persisted_transfer_reply')
        answer = json.loads(out)
        carrier.need(all(answer.get(k)==v for k,v in packet.items()) and answer['passed'] is True
                     and answer['verified_objects']==79 and answer['actual_GPU']==0
                     and answer['verified_manifest_bytes']==carrier.EXPECTED_BYTES,'independent_return_identity')
        result.update(outcome='ASSETS_REHASHED_ON_TARGET',response=answer)
    except (OSError,ValueError,KeyError,queue.Empty,subprocess.SubprocessError) as error:
        result['error_type'] = type(error).__name__
        result['outcome'] = ('UNKNOWN' if result['payload_send_attempted'] else 'PAYLOAD_NOT_SENT')
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    result['stdout_pin'] = carrier.pin(out)
    result['stderr_pin'] = carrier.pin(err)
    terminal = carrier.write(destination/'receipt.json',carrier.encode(result))
    print(json.dumps({'outcome':result['outcome'],'request':packet['request'],'response':result['response'],
                      'receipt_pin':terminal,'local_elapsed_recorded_epoch':time.time()},sort_keys=True),flush=True)
    return 0 if result['outcome']=='ASSETS_REHASHED_ON_TARGET' else 2


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--verify',action='store_true')
    cli.add_argument('--installed')
    cli.add_argument('--publication')
    options = cli.parse_args()
    if not options.verify:
        carrier.need(options.installed is not None and options.publication is not None, 'explicit_install_and_publication')
    payload = make_tar()
    tar_pin = carrier.pin(payload)
    if options.verify:
        print(json.dumps(dict(selected=79,selected_bytes=carrier.EXPECTED_BYTES,tar=tar_pin,network_operations=0)))
    else:
        destination = REPO/'local-only'/('vggt-household-selected-transfer-'+carrier.TRANSFER)
        destination.mkdir(mode=0o700)
        packet = dict(request=carrier.TRANSFER,nonce=uuid4().hex,publication=options.publication,input_SHA256=carrier.INPUT_SHA,
                      tar_SHA256=tar_pin['SHA256'],bytes=tar_pin['bytes'],installed_SHA256=options.installed)
        raise SystemExit(communicate(payload,packet,destination))
