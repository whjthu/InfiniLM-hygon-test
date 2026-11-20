from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest

from build_strict_deep_service_client import CONTRACT, REPO, materialize


PROTOCOL = REPO / 'code/bw1000/Deep-fixed-fullpre-experimental-profile-r454/source-v1/public_deep_model/protocol.py'


def compact(value):
    return json.dumps(value, separators=(',', ':')).encode()


def fixture_stream(index):
    request_id = f'fixture-{index}'

    def data(delta, finish=None):
        return b'data: ' + compact(dict(id=request_id, choices=[dict(delta=delta, finish_reason=finish)])) + b'\n\n'

    def observe(**row):
        return b': infinilm-token-clock ' + compact(dict(request_id=request_id, **row)) + b'\n\n'

    tokens = b''.join(observe(ordinal=i, token_id=token, eos=i == 2, special=i == 2, visible=i < 2)
                      for i, token in enumerate((7, 8, 1)))
    terminal = observe(terminal=True, emitted=3, notices_sent=3, observer_overflow=False,
                       owned_finished=True, finish_reason='eos_token')
    body = data(dict(role='assistant')) + tokens + data(dict(content='#### 3'), 'stop') + terminal
    return body + (b'data: [DONE]\n\n' if index != 0 else b'')


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        index = json.loads(raw)['index']
        status, body = (429, b'queue full') if index == 3 else (200, fixture_stream(index))
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream' if status == 200 else 'text/plain')
        self.send_header('Content-Length', str(len(body) + (index == 1)))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, *_):
        pass


class StrictServiceClientTest(unittest.TestCase):
    def test_generated_client_cli_on_real_http_preserves_all_twenty_requests(self):
        with TemporaryDirectory() as tmp, ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler) as server:
            root = Path(tmp)
            package = root / 'client'
            receipt = materialize(REPO, package)
            repeat = materialize(REPO, root / 'repeat')
            self.assertEqual(receipt, repeat)
            for name, digest in receipt['files_SHA256'].items():
                self.assertEqual(sha256((package / name).read_bytes()).hexdigest(), digest)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.run_client(root, package, server.server_port)
            finally:
                server.shutdown()
                thread.join(5)

    def run_client(self, root, package, port):
        module = root / 'public_deep_model'
        module.mkdir()
        raw = PROTOCOL.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), 'e4547ddf39464b2578ec6a889701727a0a591c6987876c78cf601ea66681fc46')
        (module / 'protocol.py').write_bytes(raw)
        native = 'a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
        server_cfg = dict(run='fixture-run', runtime_directory=str(root / 'runtime'), actual_step='local-fixture',
                          source_manifest_SHA256='fixture', provider_config_SHA256='fixture', gpu_start_epoch=time.time(),
                          Native_seconds=60)
        server_file = root / 'server.json'
        server_file.write_bytes(compact(server_cfg))
        run = runpy.run_path(str(PROTOCOL))['Run'](server_cfg)
        run.write('MODEL_READY.json', dict(Model_READY=True, Native_internal_owner_count=8, HC_mode=0,
                  Native_SHA256=native, fixed_profile=dict(fixed_mode=0, r366_fixed=0, HC_groups=86),
                  queue_policy=dict(enabled=True)))
        ordinals = [0, 1, 128, 129]
        records = [dict(case_id=str(i), gold=dict(suite='gsm8k', gold_numeric='3')) for i in range(4)]
        rows = [dict(index=i, case_id=str(i % 4), ordinal=ordinals[i % 4], body=dict(index=i),
                     actual_body_SHA256=sha256(compact(dict(index=i))).hexdigest(), prompt_IDs_SHA256='fixture',
                     arrival_offset_s=i * 0.01) for i in range(20)]
        plan_file = root / 'plan.json'
        plan_file.write_bytes(compact(dict(records=records, requests=rows, ordinals=ordinals)))
        config = root / 'client.json'
        out = root / 'results'
        config.write_bytes(compact(dict(private_plan=str(plan_file), private_plan_SHA256=sha256(plan_file.read_bytes()).hexdigest(),
            base_url=f'http://127.0.0.1:{port}', output=str(out), pair='A', queue_policy=True,
            deadline_unix=time.time()+30, server_config=str(server_file),
            server_config_SHA256=sha256(server_file.read_bytes()).hexdigest())))
        result = subprocess.run([sys.executable, '-B', str(package/'finite_client.py'), '--config', str(config)],
                                env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, timeout=35)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        report = json.loads((out/'result.json').read_bytes())
        self.assertEqual(report['clock_contract'], CONTRACT)
        self.assertTrue(report['completed'] and run.stop_requested())
        self.assertEqual((report['planned'], report['sent'], report['terminal'], report['incomplete']), (20, 20, 20, 3))
        self.assertEqual(report['quality_summary']['correct'], 17)
        self.assertEqual(report['quality_summary']['natural_EOS'], 17)
        cut, short, complete, rejected = report['rows'][:4]
        for failed in (cut, short):
            self.assertTrue(failed['score']['answer_matches_gold'])
            self.assertFalse(failed['score']['natural_EOS'] or failed['score']['correct'] or failed['completed'])
        self.assertFalse(cut['clock']['standard_done_received'])
        self.assertTrue(short['clock']['standard_done_received'] and short['clock']['http_body_truncated'])
        self.assertTrue(complete['completed'] and complete['score']['correct'])
        self.assertTrue(rejected['rejected'] and rejected['incomplete'])


if __name__ == '__main__':
    unittest.main()
