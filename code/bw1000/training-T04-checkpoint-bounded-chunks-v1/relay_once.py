"""One foreground finite phase. Ordinary SSH only; no local payload files/retry loop."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import runpy
import shlex
import struct
import subprocess
import threading
import time

D = Path(__file__).resolve().parent
G = D.parents[2]
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H = lambda raw: hashlib.sha256(raw).hexdigest()
BOOT = ("import sys\ndef x(n):\n b=bytearray()\n while len(b)<n:\n  q=sys.stdin.buffer.read(n-len(b))\n  if not q: raise EOFError('script truncated')\n  b.extend(q)\n return bytes(b)\n"
        "n=int.from_bytes(x(8),'little')\nif not 0<n<=262144: raise ValueError('script bound')\nexec(compile(x(n),'published-T04-chunk','exec'))\n")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def save(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def transport_state(ready, forced, rc):
    return 'RETURNED' if ready and not forced and rc not in (None, 255) else 'UNKNOWN'


def terminal_matches(value, acknowledged):
    for key in ('request_id', 'publication', 'source_manifest_SHA256', 'asset_plan_SHA256', 'endpoint_SHA256',
                'phase', 'epoch', 'deadline', 'mode', 'host', 'pid', 'startticks', 'boot_id', 'uid'):
        require(value[key] == acknowledged[key], 'terminal original endpoint identity differs')


def remaining_seconds(epoch, mono, seconds, now_epoch, now_mono, reserve=15):
    value = min(epoch+seconds-now_epoch, mono+seconds-now_mono)-reserve
    if value <= 0:
        raise TimeoutError('original phase deadline exhausted')
    return value


def kill_children(children, forced, errors):
    for child in list(children.values()):
        try:
            if child.poll() is None:
                forced.add(child.pid)
                child.kill()
        except Exception as error:
            # A child exiting concurrently must not skip the other owned child.
            errors.append('owned-child-kill:'+type(error).__name__)


def main():
    epoch, mono = time.time(), time.monotonic()  # Admission/Git are charged to this original epoch.
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--nonce', required=True)
    args = parser.parse_args()
    seconds = 90 if args.phase == 0 else 7200

    def left(reserve=15):
        return remaining_seconds(epoch, mono, seconds, time.time(), time.monotonic(), reserve)

    require(re.fullmatch('[0-9a-f]{40}', args.publication) and re.fullmatch('[0-9a-f]{32}', args.nonce), 'publication/nonce invalid')
    mf = (D/'SOURCE_FILES.json').read_bytes()
    require(H(mf) == args.source_manifest_sha, 'local source MF differs')
    git = ['git', '-c', 'safe.directory='+G.as_posix(), '-C', str(G)]
    require(subprocess.check_output(git+['show', args.publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json'], timeout=min(12, left())) == mf,
            'source not published exact bytes')
    blobs = {}
    for name, pin in json.loads(mf)['files'].items():
        raw = (D/name).read_bytes()
        require(len(raw) == pin['bytes'] and H(raw) == pin['SHA256'], 'source member differs')
        blobs[name] = raw
    plan, claim = json.loads(blobs['PLAN.json']), json.loads(blobs['CLAIM.json'])
    require(seconds == (plan['phase0_epoch_seconds'] if args.phase == 0 else plan['epoch_seconds']), 'frozen phase duration differs')
    require(claim['root_GO'] is True and claim['transfer_GO'] is True and claim['case'] == 'T04.sft'
            and args.phase in claim['allowed_phases'] and claim['max_invocations_each_phase'] == 1, 'phase not approved')
    for pin in (plan['source_authority'], plan['metadata_terminal']):
        raw = subprocess.check_output(git+['show', args.publication+':'+pin['path']], timeout=min(12, left()))
        require(H(raw) == pin['SHA256'] and ('bytes' not in pin or len(raw) == pin['bytes']), 'published dependency differs')
    if args.phase > 0:
        # Independent consumption is a prerequisite, not inferred from a producer's PASS.
        pin = claim['prior_phase_terminal_authority']
        require(isinstance(pin, dict), 'prior original phase independently consumed authority missing')
        raw = subprocess.check_output(git+['show', args.publication+':'+pin['path']], timeout=min(12, left()))
        require(len(raw) == pin['bytes'] <= 65536 and H(raw) == pin['SHA256'], 'published prior authority differs')
        prior = json.loads(raw)
        old_intent = json.loads((G/'local-only'/plan['label']/('%02d' % (args.phase-1))/'intent.json').read_bytes())
        require(prior['case'] == 'T04.sft' and prior['phase'] == args.phase-1
                and prior['request_id'] == old_intent['request_id']
                and prior['source_independently_verified'] is True and prior['receiver_independently_verified'] is True
                and prior['same_original_host_processes_absent'] is True
                and prior['asset_plan_SHA256'] == H(blobs['PLAN.json'])
                and prior['endpoint_SHA256'] == H(blobs['endpoint.py']), 'prior original request UNKNOWN/not fully consumed')
    phase = plan['phases'][args.phase]
    require(phase['id'] == args.phase and phase['payload_bytes'] <= 1<<30, 'phase byte cap differs')
    require(H(HELPER.read_bytes()) == HELPER_SHA, 'existing SSH helper differs')
    ssh = runpy.run_path(str(HELPER))['ssh']
    left()
    root = G/'local-only'/plan['label']
    root.mkdir(exist_ok=True)
    local = root/('%02d' % args.phase)
    local.mkdir()  # fixed per phase, even a different nonce cannot replay an existing invocation
    context = dict(publication=args.publication, source_manifest_SHA256=args.source_manifest_sha,
                   asset_plan_SHA256=H(blobs['PLAN.json']), endpoint_SHA256=H(blobs['endpoint.py']),
                   request_id=args.nonce, phase=args.phase, epoch=epoch, deadline=epoch+seconds)
    save(local/'intent.json', context)
    children, ready, forced, errors, readers = {}, {}, set(), [], []
    forwarded = 0

    def kill_owned():
        kill_children(children, forced, errors)

    timer = threading.Timer(left(), kill_owned)
    timer.daemon = True; timer.start()

    def line(stream, bound):
        result = queue.Queue(maxsize=1)
        threading.Thread(target=lambda: result.put(stream.readline(bound+1)), daemon=True).start()
        raw = result.get(timeout=left())
        require(0 < len(raw) <= bound and raw.endswith(b'\n'), 'control record truncated/oversized')
        return json.loads(raw)

    def start(mode):
        code = ('PLAN='+repr(plan)+'\nCONTEXT='+repr(context)+'\nMODE='+repr(mode)+'\n').encode()+blobs['endpoint.py']
        require(0 < len(code) <= 262144, 'bootstrap script too large')
        command = ('/home/whj/nvidia-baseline/env-vlm/bin/python' if mode == 'source' else '/usr/local/python3.12/bin/python3')+' -I -S -B -c '+shlex.quote(BOOT)
        if mode == 'source':
            argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
                    '-F', 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config', 'a100-27', command]
        else:
            argv = [ssh[0], '-o', 'StrictHostKeyChecking=yes', *ssh[1:], command]
        require(len(subprocess.list2cmdline(argv)) < 30000, 'Win32 argv bound')
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        children[mode] = child
        left()  # Timer may have fired in Popen before this held child was registered.
        save(local/(mode+'-owned-ssh.json'), dict(pid=child.pid, held_unreaped=True, deadline=context['deadline']))

        def drain():
            used, digest = 0, hashlib.sha256()
            try:
                with (local/(mode+'-stderr.private')).open('xb') as out:
                    while True:
                        block = child.stderr.read(65536)
                        if not block:
                            break
                        used += len(block)
                        require(used <= 262144, 'stderr bound exceeded')
                        digest.update(block); out.write(block)
            except BaseException as error:
                errors.append(type(error).__name__); kill_owned()
            finally:
                save(local/(mode+'-stderr-pin.json'), dict(bytes=used, SHA256=digest.hexdigest()))
        thread = threading.Thread(target=drain, daemon=True); thread.start(); readers.append(thread)
        left()
        child.stdin.write(struct.pack('<Q', len(code))+code); child.stdin.flush()
        if mode == 'source':
            child.stdin.close()
        value = line(child.stdout, 8192)
        save(local/(mode+'-first-control.json'), value)
        require(value['stage'] == 'READY' and value['mode'] == mode and all(value[k] == v for k, v in context.items()), 'READY context differs')
        require(value['host'] and value['boot_id'] and value['pid'] > 0 and value['startticks'] > 0
                and value['uid'] == {'source': 1001, 'receiver': 154459}[mode]
                and value['payload_bytes'] == phase['payload_bytes'] and len(value['CPU_affinity']) == 2
                and value['AS_bytes'] == plan['AS_bytes'] and value['actual_GPU_cards'] == 0, 'READY ownership/resources differ')
        ready[mode] = value
        save(local/(mode+'-remote-owned.json'), value)
        return child

    outcome = dict(context, passed=False, forwarded_payload_bytes=0, replay_allowed=False, GPU_GO=False,
                   source='NOT_DISPATCHED', receiver='NOT_DISPATCHED')
    try:
        receiver = start('receiver')
        source = start('source')
        while forwarded < phase['payload_bytes']:
            left()
            block = source.stdout.read1(min(plan['bridge_block_bytes'], phase['payload_bytes']-forwarded))
            require(block, 'source payload truncated')
            require(receiver.stdin.write(block) == len(block), 'bridge short write')
            forwarded += len(block)
        source_receipt = line(source.stdout, 65536)
        terminal_matches(source_receipt, ready['source'])
        source.wait(timeout=left())
        require(source.returncode == 0 and source_receipt['passed'] is True, 'source phase failed')
        save(local/'source-validation.json', source_receipt)
        terminal = (json.dumps(source_receipt, sort_keys=True, separators=(',', ':'))+'\n').encode()
        receiver.stdin.write(terminal); receiver.stdin.close()
        target_receipt = line(receiver.stdout, 65536)
        terminal_matches(target_receipt, ready['receiver'])
        receiver.wait(timeout=left())
        save(local/'receiver-validation.json', target_receipt)
        require(receiver.returncode == 0 and target_receipt['passed'] is True
                and target_receipt['segments'] == source_receipt['segments'], 'receiver phase failed')
        outcome['passed'] = True
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error_reason=str(error)[:180])
    finally:
        kill_owned()
        for mode, child in children.items():
            try:
                remaining = max(.001, min(context['deadline']-time.time(), mono+seconds-time.monotonic()))
                child.wait(timeout=min(4, remaining))
            except subprocess.TimeoutExpired:
                forced.add(child.pid)
            outcome[mode] = transport_state(mode in ready, child.pid in forced, child.returncode)
            outcome[mode+'_transport'] = dict(returncode=child.returncode, forced_local_SSH_kill=child.pid in forced,
                remote_outcome='UNKNOWN_RECONCILE_EXACT_NONCE' if outcome[mode] == 'UNKNOWN' else 'RETURNED_REQUIRES_INDEPENDENT_RECEIPT_AUDIT')
        timer.cancel()
        for thread in readers:
            remaining = max(0, min(context['deadline']-time.time(), mono+seconds-time.monotonic()))
            thread.join(timeout=min(1, remaining))
    outcome.update(forwarded_payload_bytes=forwarded, local_payload_files_written=0,
                   payload_buffer_bound_bytes=plan['bridge_block_bytes'], forward_RAM_cap_bytes=8<<20,
                   elapsed_seconds=time.monotonic()-mono, stream_errors=errors, full_asset_qualified=False)
    outcome['passed'] = bool(outcome['passed'] and not errors and not forced and time.time() <= context['deadline'])
    save(local/'outcome.json', outcome)
    print(json.dumps(outcome, allow_nan=False))
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
