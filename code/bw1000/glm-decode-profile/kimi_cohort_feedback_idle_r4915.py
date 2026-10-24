"""Bounded CPU feedback transport and sequential-owner PP decode scheduler.

Experimental idle/abort candidate: phase idle reads have a separate deadline.
Case dependencies keep their own short failstop timeout. CPU failures wake receivers.
No native GPU/CCL abort or bounded process teardown is proven.
No GPU speed or model equivalence is established by this file.
The engine owner makes every forward synchronously; socket handlers never touch
engine state. A cohort receives its preceding token before its next forward.
"""
import copy
import json
import math
import socket
import socketserver
import threading
import time

MAX_FRAME = 1024 * 1024


def _send(sock, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_FRAME:
        raise ValueError("feedback frame exceeds bound")
    sock.sendall(raw)


def _read(reader):
    raw = reader.readline(MAX_FRAME + 1)
    if not raw or len(raw) > MAX_FRAME or not raw.endswith(b"\n"):
        raise ValueError("EOF, oversized or incomplete feedback frame")
    return json.loads(raw)


class FeedbackHub:
    def __init__(self, host, port, nonce, participants=4, emitter=3, timeout=120, idle_timeout=7200):
        assert nonce and 1 < participants <= 128 and 0 <= emitter < participants
        assert 0 < timeout <= 7200
        self.nonce, self.participants, self.emitter = nonce, participants, emitter
        assert timeout + 5 < idle_timeout <= 21600
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.cv = threading.Condition()
        self.case = None
        self.pending = {}
        self.closed = False
        self.failure = None
        self.active_peers = set()
        self.stats = {}
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(outer.idle_timeout)
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                rank = None
                registered = False
                try:
                    hello = _read(self.rfile)
                    rank = hello.get("rank")
                    assert hello.get("nonce") == outer.nonce, "nonce mismatch"
                    assert type(rank) is int and 0 <= rank < outer.participants
                    with outer.cv:
                        assert rank not in outer.active_peers, "duplicate live feedback rank"
                        outer.active_peers.add(rank)
                        registered = True
                        outer.cv.notify_all()
                    _send(self.request, {"ok": True})
                    while True:
                        request = _read(self.rfile)
                        try:
                            value = outer.dispatch(rank, request)
                            _send(self.request, {"ok": True, "value": value})
                        except Exception as error:
                            outer.fail_case("rank %d protocol failure: %s" % (rank, str(error)))
                            _send(self.request, {"ok": False, "error": str(error)})
                except (OSError, ValueError, AssertionError):
                    return
                finally:
                    if registered:
                        with outer.cv:
                            outer.active_peers.remove(rank)
                            outer.fail_case("rank %d feedback connection lost" % rank)
                            outer.cv.notify_all()

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = False
            daemon_threads = True
            block_on_close = False

        self.server = Server((host, port), Handler)
        self.address = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def fail_case(self, message):
        # Terminal latch: a failed case cannot silently resume with stale tokens.
        with self.cv:
            if self.case is not None and not self.closed and self.failure is None:
                self.failure = message
            self.cv.notify_all()

    def _check_live(self):
        if self.failure is not None:
            raise RuntimeError("feedback case failed: " + self.failure)
        if self.closed:
            raise RuntimeError("feedback hub closed")

    def _wait(self, predicate):
        deadline = time.monotonic() + self.timeout
        while True:
            self._check_live()
            if predicate():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("feedback dependency timed out")
            self.cv.wait(remaining)

    def dispatch(self, rank, req):
        with self.cv:
            self._check_live()
            op = req["op"]
            if op == "begin":
                assert rank == self.emitter and self.case is None and not self.pending
                cohorts = req["cohorts"]
                assert cohorts and all(c and all(type(v) is int and v >= 0 for v in c) for c in cohorts)
                flat = sum(cohorts, [])
                assert len(flat) == len(set(flat)) and len(flat) <= 128
                assert isinstance(req["case"], str) and 0 < len(req["case"]) <= 240
                self._wait(lambda: self.active_peers == set(range(self.participants)))
                self.case, self.cohorts = req["case"], copy.deepcopy(cohorts)
                self.publish_next = [0] * len(cohorts)
                self.take_next = {r: [0] * len(cohorts) for r in range(self.participants) if r != self.emitter}
                self.stats = {"published": 0, "takes": 0, "peak_pending": 0,
                              "participants": self.participants, "cohorts": len(cohorts)}
                return dict(self.stats)
            assert req["case"] == self.case and self.case is not None, "case mismatch"
            if op == "end":
                assert rank == self.emitter and not self.pending, "undrained feedback"
                assert all(v == self.publish_next for v in self.take_next.values()), "unconsumed keys"
                result = dict(self.stats)
                result["published_per_cohort"] = list(self.publish_next)
                self.case = None
                return result
            cohort, step = req["cohort"], req["step"]
            assert type(cohort) is int and 0 <= cohort < len(self.cohorts)
            assert type(step) is int and step >= 0
            key = (cohort, step)
            if op == "publish":
                assert rank == self.emitter, "only emitter publishes"
                assert step == self.publish_next[cohort], "duplicate/skipped publication"
                packet = req["packet"]
                rows, ids = packet["rows"], packet["ids"]
                assert rows and len(rows) == len(ids) and len(rows) == len(set(rows))
                assert rows == [r for r in self.cohorts[cohort] if r in rows], "cohort order changed"
                assert all(type(v) is int and v >= 0 for v in ids)
                assert all(isinstance(packet[k], (int, float)) and math.isfinite(packet[k]) and packet[k] >= 0
                           for k in ("stamp", "seconds"))
                self._wait(lambda: len(self.pending) < 2 * len(self.cohorts))
                self.pending[key] = (copy.deepcopy(packet), set(self.take_next))
                self.publish_next[cohort] += 1
                self.stats["published"] += 1
                self.stats["peak_pending"] = max(self.stats["peak_pending"], len(self.pending))
                self.cv.notify_all()
                return None
            assert op == "take" and rank != self.emitter, "invalid receiver operation"
            assert step == self.take_next[rank][cohort], "duplicate/skipped dependency"
            self._wait(lambda: key in self.pending)
            packet, readers = self.pending[key]
            assert rank in readers
            readers.remove(rank)
            self.take_next[rank][cohort] += 1
            self.stats["takes"] += 1
            if not readers:
                del self.pending[key]
            self.cv.notify_all()
            return copy.deepcopy(packet)

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()


class FeedbackClient:
    def __init__(self, host, port, nonce, rank, timeout=120):
        self.socket = socket.create_connection((host, port), timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = self.socket.makefile("rb")
        try:
            _send(self.socket, {"nonce": nonce, "rank": rank})
            assert _read(self.reader) == {"ok": True}
        except BaseException:
            self.close()
            raise

    def request(self, op, case, **values):
        _send(self.socket, {"op": op, "case": case, **values})
        response = _read(self.reader)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["value"]

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.close()
        self.socket.close()


def decode_cohorts(*, feedback, case, is_emitter, ids, pasts, times,
                   micro, limit, eos, forward, clock=time.perf_counter, anchor=None):
    """Modify serving rows in place; preserve cohort membership after EOS.

    Caller begins feedback case and establishes the common anchor first. The
    forward callback is synchronous, returns (tail_ids_or_None, seconds), and
    must not globally collect. Timestamp origin is local; only emitter stamps
    become canonical. Mixed EOS is a separately qualified geometry because the
    legacy scheduler compacts rows across cohorts.
    """
    assert len(ids) == len(pasts) == len(times) and micro > 0 and limit > 0
    cohorts = [list(range(a, min(a + micro, len(ids)))) for a in range(0, len(ids), micro)]
    outstanding = [None] * len(cohorts)
    next_step = [0] * len(cohorts)
    samples = []
    start = clock() if anchor is None else anchor

    def accept(packet, rows):
        assert packet["rows"] == rows and len(rows) == len(packet["ids"])
        for row, value in zip(rows, packet["ids"]):
            assert len(ids[row]) < limit and ids[row][-1] not in eos
            assert type(value) is int and value >= 0
            ids[row].append(value)
            pasts[row] += 1
            times[row].append(packet["stamp"])

    while True:
        launched = False
        for cohort, members in enumerate(cohorts):
            if outstanding[cohort] is not None:
                step, previous_rows = outstanding[cohort]
                packet = feedback.request("take", case, cohort=cohort, step=step)
                accept(packet, previous_rows)
                outstanding[cohort] = None
            rows = [i for i in members if len(ids[i]) < limit and ids[i][-1] not in eos]
            if not rows:
                continue
            step = next_step[cohort]
            selected, seconds = forward(rows, step, cohort)
            samples.append(seconds)
            if is_emitter:
                assert selected is not None
                packet = {"rows": rows, "ids": selected, "stamp": clock() - start, "seconds": seconds}
                feedback.request("publish", case, cohort=cohort, step=step, packet=packet)
                accept(packet, rows)
            else:
                assert selected is None
                outstanding[cohort] = (step, rows)
            next_step[cohort] += 1
            launched = True
        if not launched:
            break
    assert all(v is None for v in outstanding)
    return {"samples": samples, "forwards_per_cohort": next_step,
            "cohorts": cohorts, "local_wall_seconds": clock() - start}
