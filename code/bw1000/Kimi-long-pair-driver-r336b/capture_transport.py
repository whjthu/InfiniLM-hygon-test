"""One bounded node2->node0 five-tap harvest stream, outside target WORLD."""
import socket
from batch_wire_r9128 import send,recv
from publisher import NATIVE_SHA


def establish(native,head,port,nonce,timeout):
    if type(nonce)is not str or len(nonce)<32 or not 1<=timeout<=7200:
        raise ValueError('Explicit bounded capture transport identity')
    listener=peer=None
    hello=dict(node=2,nonce=nonce,Native_SHA256=NATIVE_SHA,protocol='Kimi-long-capture-r336')
    try:
        if native.node==0:
            listener=socket.socket();listener.settimeout(timeout)
            listener.bind((head,port));listener.listen(1)
        packets=native.control.collect('long-capture-listener',dict(node=native.node,ready=True))
        if len(packets)!=4 or {p['node']for p in packets}!=set(range(4)):
            raise ValueError('Exact four capture participants')
        if native.node==2:
            peer=socket.create_connection((head,port),timeout=timeout)
            peer.settimeout(timeout);send(peer,hello)
            reply,body=recv(peer)
            if reply!=dict(ready=True,protocol='Kimi-long-capture-r336') or body:
                raise ValueError('Capture receiver identity mismatch')
            native.peer=peer
        elif native.node==0:
            peer,_=listener.accept();peer.settimeout(timeout)
            packet,body=recv(peer)
            if packet!=hello or body: raise ValueError('Capture sender identity mismatch')
            send(peer,dict(ready=True,protocol='Kimi-long-capture-r336'))
            native.peers={2:peer}
        native.control.collect('long-capture-connected',dict(node=native.node,connected=True))
        return peer
    except BaseException:
        if peer is not None:peer.close()
        raise
    finally:
        if listener is not None:listener.close()


def close(native):
    peers=[native.peer]+list(native.peers.values())
    native.peer=None;native.peers={}
    for peer in peers:
        if peer is not None:peer.close()
