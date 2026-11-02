"""Default OFF. ON has a distinct authenticated policy/source identity."""
import re

OFF='off'
ON='projected-only-r384'
CONTEXT_SHA256='d5b9f9ee522af81ef5b062b06aebffc8fb5a1b8055dd3c942dda4f377d8d967c'
PROTOCOL_ON='Kimi-C1-long-context-r388'
OLD_WORKER_SHA256='813c274a3204e09ba23f7715baed5b8943985625c4c5ac341e497bea5c2763f3'
OLD_MANIFEST_SHA256='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31'

def checked(mode):
 if type(mode)is not str or mode not in [OFF,ON]:raise ValueError('Explicit supported context policy required')
 return mode

def expected_hello(old,nonce,mode=OFF,worker_manifest_SHA256=None):
 checked(mode);hello=old.expected_hello(nonce)
 if mode==OFF:return hello
 if not isinstance(worker_manifest_SHA256,str)or not re.fullmatch('[0-9a-f]{64}',worker_manifest_SHA256):raise ValueError('ON requires exact private worker manifest')
 return dict(hello,protocol=PROTOCOL_ON,context_policy=ON,context_source_SHA256=CONTEXT_SHA256,worker_entry_manifest_SHA256=worker_manifest_SHA256)

def ready(old,mode,hello):
 value=dict(ready=True,protocol=old.PROTOCOL,C=1,context=32768,block_size=8,proposals=7,prompt_append_max=512,committed_append_max=8,runtime_manifest_SHA256=old.RUNTIME_SHA,target_Native_SHA256=old.TARGET_NATIVE_SHA)
 if mode==OFF:return value
 return dict(value,protocol=PROTOCOL_ON,context_policy=ON,context_source_SHA256=CONTEXT_SHA256,worker_entry_manifest_SHA256=hello['worker_entry_manifest_SHA256'])
