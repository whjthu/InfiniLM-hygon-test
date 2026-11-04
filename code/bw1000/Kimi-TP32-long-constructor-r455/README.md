This is a TP32PP1 **constructor-only consumer**, not a runnable device launcher.
It accepts the already provider-verified r453 library and checkpoint config/index
metadata. It creates once, observes the actual exported geometry/keys/head
records, and destroys on the same caller thread, including on inspection failure.
It never opens safetensors, loads/finishes weights, resets, forwards, or starts HTTP.

Current CPU961 DSO is **STOP_CPU961_ALLOCATION**: exact I96 regular-pool W2
allocation plus strongly resident expert banks and replicated BF16 children
already exceed actual visible VRAM minus the original 4 GiB floor, before
cache/head/attention. `CAPACITY_STOP.json` pins that separate source/binary/585
raw audit and independently recomputes its decisive integer lower bound.
No CPU job, keeper, or GPU has been submitted by r455.
The 4 GiB floor and full 32768 capacity must remain unchanged. This directory
does not include an executable device command or a GPU claim.

The APIs expose actual per-node cache configuration and per-owner head tensors.
They do not expose per-owner cache allocations or per-layer MXFP4 scale tensor
addresses/allocated bytes. Those remain source-derived until a suitable actual
observation exists. Capability fields are not independent allocator measurements.
Affinity is retained as a raw snapshot: the old strict post-first-submit check
cannot be asserted before a genuine owned submission. No dummy forward or manual
thread binding is introduced to manufacture changes==1.

`check_local.py` is a new, local ctypes/libffi high-pointer and exception-cleanup
consumer. Its callbacks record Native leaves; it is not a device or Native test.
The r453 C++ source, objects, DSO, and CPU961 evidence remain immutable.
