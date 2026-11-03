# Private strict-F32 value leaves, default OFF

This is an implemented source candidate, not a compiled/device-qualified result.
No CPU job, GPU, Model load, shared SDK change or original1368 replay was run.

r443 proved that the installed Hygon SDK rejects config-aware FullFp32 GEMM;
Cast alone cannot close r433. This package replaces **both** rejected interfaces
inside the private r433 component:3 software Casts and16 fixed F32 GEMV leaves.
It never changes the SDK Config, selects DefaultFast, or borrows an unowned cache.
The32/64-bit scalar/pointer CABI and original owned Runtime/device/stream methods
are used directly. No new Runtime/class-layout/provider ABI is required.

The fixed GPU implementation is B256, one output/thread, ascending-K `fmaf`:
8 aggregations [1,2048]@[2048,512] and8 projections [1,512]@[512,256]. Products,
accumulators and outputs are F32. There is no splitK, reduction grid, TF32,
matrix mixed precision, intermediate BF16 aggregate, or GPU F64. This deliberately
small kernel may underutilize the device; its speed is UNKNOWN. The reduction
association can differ from vendor GEMM; it is not a bitexact equivalence claim.

Cast is bit expansion BF16->F32 and explicit integer RN-even F32->BF16. Signed
zero, finite subnormals, midpoint ties and BF16 overflow are explicit. NaN/Inf
input or output sets an owned device flag. The complete query uses one completion-returning Runtime D2H of
the4-byte status and refuses nonfinite results. Successful leaf destruction has
no extra sync; only unfinished/error destruction drains. Original enclosing
attention/Work fences remain. New IEEE/no-fast-math
compiler flags, scalar FMA ISA and denormal mode still need actual compile/device
qualification. Unsupported flags must fail the planned CPU build, never be dropped.

Unchanged r433 mathematical path: BF16 QK256; original scale/BF16 rounding;
F32 softmax followed by BF16 probabilities; actual same per-row FP8 quant;
existing identity-r776 F32 roundtrip; existing owned W dequant setup; the full
global128 weight scales and8 interleaved448-row heads (V starts192) are retained.
No Wv contiguous-slice shortcut or aggregate requantization is introduced.
The original finite-precision value reassociation remains approximate versus
cached/recomputed per-row BF16 V. Task quality and complete attention gain remain
UNKNOWN; r431 diagnostic failure and r433 unsupported result remain unchanged.

`Owned` checks the same Runtime/owner/stream/device/eager state. The status tensor
and host staging survive all enqueues and error drains. The enclosing original
Work holds all input/intermediate/output tensors, and prepared V remains owned.
No destructor releases stream-used memory before drain. All3 casts,16 launches,
temporary/status allocation, H2D/D2H/sync/flag checks and destruction are inside
the original complete CAPI wall. One-time identity/W setup and4MiB persistent V
are reported separately and in the cold call, as in r433. The original row
roundtrip (~1.074GF) and setup (~1.879GF) are still real costs; not free dequant.

The complete parent Context owns a real8-rank communicator and O-projection
allreduce. A1-card kernel test cannot be called this full component. Any future
complete comparison should retain the original8-card attention CAPI, real TP8,
checkpoint inputs/selection geometry, state seed/restore, output projection,
normal8 Runtime destruction, and independent post check. Root identified910374
n07/8cards<=600s as a possible future scope; no GPU claim is activated here.

Prepared CPU scope: one new HOST consumer +one new gfx936 GPU object,29 exact
parent objects reused (31 final), standalone pure bitfield/F32 reference, MD and
source-byte pin checks, original strong/CABI exports, exact dynamic providers.
The standalone executable links no SDK, so it does not hit the static SDK GPU
constructor boundary. The linked component is inspected, never dlopened on CPU.
Original872/879 suites,29 objects, SDK and helper are not recompiled/rerun.

The new explicit `glm_attention_value_fullfp32_r446(void*,int)` selector forwards
the unchanged clean-NoV-epoch/owner check; default remains OFF. A future Python
consumer must declare `[ctypes.c_void_p,ctypes.c_int]`, `restype=ctypes.c_int` and
verify actual symbol provider/hash before its first call. Counters report
GEMV leaf calls, not SDK GEMM submits or inferred GPU kernel counts.
