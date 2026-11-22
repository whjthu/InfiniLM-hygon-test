"""Fixed pinned transfer buffers with independent full-payload and guard verification."""

import ctypes as C
from hashlib import sha256
import random

from contract import TRANSFER_BYTES, need


class Transfer:
    def __init__(self, hip, dma, resources, seed):
        self.hip, self.dma = hip,dma
        self.host = dma.new_buffer(hip,resources,True,TRANSFER_BYTES)
        self.device = dma.new_buffer(hip,resources,False,TRANSFER_BYTES)
        self.scratch = C.create_string_buffer(TRANSFER_BYTES)
        payload = random.Random(seed).randbytes(TRANSFER_BYTES)
        self.pin = sha256(payload).hexdigest()
        hip.call("Memcpy",self.device[1],payload,TRANSFER_BYTES,dma.H2D)
        for index,guard in zip((0,2),dma.GUARDS,strict=True):
            C.memmove(self.host[index],guard,dma.GUARD)
            hip.call("Memcpy",self.device[index],guard,dma.GUARD,dma.H2D)

    def poison(self):
        C.memset(self.host[1],0,TRANSFER_BYTES)

    def prepare_chunks(self):
        half = TRANSFER_BYTES // 2
        need(TRANSFER_BYTES == 2 * half and half % 256 == 0, "unaligned fixed D2H midpoint")
        destination, source = self.host[1], self.device[1]
        self.placements = (
            ((destination, source, TRANSFER_BYTES),),
            ((destination, source, half),
             (self.dma.add(destination, half), self.dma.add(source, half), half)),
        )

    def enqueue(self, stream, calls):
        for destination, source, size in self.placements[calls - 1]:
            self.hip.call("MemcpyAsync", destination, source, size, self.dma.D2H, stream)

    def audit(self):
        dma = self.dma
        need(sha256(C.string_at(self.host[1],TRANSFER_BYTES)).hexdigest()==self.pin,
             "pinned D2H payload SHA differs")
        self.hip.call("Memcpy",self.scratch,self.device[1],TRANSFER_BYTES,dma.D2H)
        need(sha256(self.scratch.raw).hexdigest()==self.pin,"immutable D2H source payload differs")
        for index,expected in zip((0,2),dma.GUARDS,strict=True):
            self.hip.call("Memcpy",self.scratch,self.device[index],dma.GUARD,dma.D2H)
            need(C.string_at(self.host[index],dma.GUARD)==expected
                 and C.string_at(self.scratch,dma.GUARD)==expected,"D2H redzone differs")
        return self.pin
