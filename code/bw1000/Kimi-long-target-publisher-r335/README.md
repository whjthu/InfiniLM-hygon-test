# C1 long target publisher

This private, default-OFF package joins the 330b Native adapter to the authenticated
334 worker protocol. It keeps the qualified greedy selection policy unchanged.
Only node 0 owns the draft connection; all four target leaders call Publisher in
the same order. The controller must bind the exact Native and worker sources,
provide bounded socket/control deadlines, and own process/resource cleanup.

Prompt handling resets with state capture OFF, seeds and appends at most 512
tokens, publishes committed captures, and seals without resetting target state.
The private adapter adds an `on_selected` callback to true-input append, so the
last prompt's token can be emitted before feature harvest and network transfer.
The emitter callback must emit only on the designated service emitter and record
its actual timeline. A later failure remains a failed partial request.

Decode queries at most seven proposals, verifies anchor plus proposals, commits
the real greedy decision, and performs state-only replay on a partial match.
All four leaders must report the expected active/shadow swap and committed past
before a receipt or cropped feature prefix is sent. Terminal requests retire
the worker context without publishing terminal features. Past 32760 uses AR
tail steps without draft queries. Frozen prompts reserve 128 output positions.

`DraftClient` takes an already connected socket and performs one exact nonce and
source handshake. It does not connect, launch, reload, retry, or reconnect. Any
publisher failure poisons the publisher and closes node 0's socket. The driver
must stop all target leaders on failure; a local exception is not cluster cleanup.

CPU checks execute the real publisher/policy and binary client framing against
recording Native/control boundaries. The adapter check invokes actual Python
adapter methods with a recording C ABI. These checks do not execute a real
Native Context, GPU capture, model checkpoint, full worker query, or target
commit/replay. No device, acceptance-rate or performance qualification follows.

Remaining before the planned four-request device run: whole AR/spec driver,
two frozen long token inputs, current qualified baseline and device mapping,
actual 32-target plus one-draft launch/deadline/cancellation/cleanup integration.
No GPU GO is created by this package.
