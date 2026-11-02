"""Own one bind-only FD before GPU work; activate that exact FD after loading."""
import socket
from lifecycle import NATIVE_SHA


class CapturePortLease:
    def __init__(self, node, head, port, check):
        if node not in range(4) or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('Exact capture node/port required')
        self.node, self.head, self.port, self.check = node, head, port, check
        self.listener = self.peer = None
        self.used = self.closed = False
        self.bound_fileno = None
        if node == 0:
            listener = socket.socket()
            try:
                listener.set_inheritable(False)
                # No reuse flags, listen, accept or connect before the GPU/weight phase.
                assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 0
                if hasattr(socket, 'SO_REUSEPORT'):
                    assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0
                listener.bind((head, port))
                self.port = listener.getsockname()[1]
                self.bound_fileno = listener.fileno()
                assert not listener.get_inheritable()
                assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 0
                self.listener = listener
            except BaseException:
                listener.close()
                raise

    def info(self):
        return dict(node=self.node, head=self.head, port=self.port,
                    bound_before_GPU=self.node == 0, same_FD_no_rebind=True,
                    listening=bool(self.listener and self.listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)),
                    inheritable=False, SO_REUSEADDR=False, SO_REUSEPORT=False,
                    nonce_or_payload_logged=False)

    def establish(self, native, nonce, timeout):
        from batch_wire_r9128 import send, recv
        if self.closed or self.used or native.node != self.node:
            raise RuntimeError('Capture lease is single-use and owned by this node')
        if not isinstance(nonce, str) or len(nonce) < 32 or not 1 <= timeout <= 7200:
            raise ValueError('Original bounded capture identity required')
        self.used = True
        peer = None
        hello = dict(node=2, nonce=nonce, Native_SHA256=NATIVE_SHA, protocol='Kimi-long-capture-r336')
        try:
            self.check('capture-listen-after-weights')
            if self.node == 0:
                if self.listener is None or self.listener.fileno() != self.bound_fileno:
                    raise RuntimeError('Original preweight bound FD was lost; no rebind')
                assert self.listener.getsockname() == (self.head, self.port)
                self.listener.settimeout(timeout)
                self.listener.listen(1)
            packets = native.control.collect('long-capture-listener', dict(node=self.node, ready=True))
            if len(packets) != 4 or {p['node'] for p in packets} != set(range(4)):
                raise ValueError('Exact four capture participants required')
            if self.node == 2:
                peer = socket.create_connection((self.head, self.port), timeout=timeout)
                peer.settimeout(timeout); send(peer, hello)
                reply, body = recv(peer)
                if reply != dict(ready=True, protocol='Kimi-long-capture-r336') or body:
                    raise ValueError('Capture receiver identity mismatch')
                native.peer = peer
            elif self.node == 0:
                peer, _ = self.listener.accept(); peer.settimeout(timeout)
                packet, body = recv(peer)
                if packet != hello or body: raise ValueError('Capture sender identity mismatch')
                send(peer, dict(ready=True, protocol='Kimi-long-capture-r336'))
                native.peers = {2: peer}
            self.peer = peer
            native.control.collect('long-capture-connected', dict(node=self.node, connected=True))
            return peer
        except BaseException:
            if peer is not None: peer.close()
            raise
        finally:
            if self.listener is not None:
                self.listener.close(); self.listener = None

    def close(self, native=None, *, normal=False):
        """Always release ownership; normal sender closes first to avoid listener TIME_WAIT."""
        error = None
        try:
            if self.peer is not None and normal:
                self.peer.settimeout(2)
                try:
                    if self.node == 2: self.peer.shutdown(socket.SHUT_WR)
                    # No residual frames are permitted after successful final harvest.
                    if self.peer.recv(1) != b'':
                        raise ValueError('Residual capture payload at normal close')
                except BaseException as exc:
                    error = exc
        finally:
            if self.peer is not None: self.peer.close()
            if self.listener is not None: self.listener.close()
            self.peer = self.listener = None; self.closed = True
            if native is not None:
                native.peer = None; native.peers = {}
        if error is not None: raise error
