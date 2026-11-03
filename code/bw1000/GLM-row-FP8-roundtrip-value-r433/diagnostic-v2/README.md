# GLM r433 opt-in diagnostic source

Adds four owner-only bounded CAPI functions to retain actual query-produced Runtime tensors, describe them, copy exact typed bytes after stream quiescence, and clear. Default OFF. Two tickets cover OFF/ON firstwarm of one C1/U2048 fixture. rank0 can save full math inputs/intermediates; other ranks expose actual outputs. Formal queries do not arm snapshots.

No raw D2H runs inside formal CAPI timing. Warm snapshots deliberately extend tensor lifetimes, so warm CAPI plus external read/clear cost is reported separately. Every completed query still needs its original CPU output bytes saved before diagnostic refusal; the device driver is not frozen by this CPU package.

A small firstON dtype2 identity selfcheck is inside cold setup; unsupported FullFp32 or identity failure stops. Original row quantization, F32 association and output BF16 math from CPU872 remain unchanged.

CPU claim compiles one changed HOST, reuses the 29 unchanged objects, links, checks fresh MD/provider bytes and only the new libffi/raw-file/error consumers. No old872 geometry/FFI suites, Native dlopen, Runtime, weights or GPU.
