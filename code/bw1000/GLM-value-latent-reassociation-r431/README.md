# Value-only latent contraction

Private default-OFF C1/U2048/layer38 attention component. Preserve residentK, QK256, original BF16 scale-round, F32 softmax and BF16 probability. Eight existing 2D GEMMs aggregate selected rawBF16 latent, then original M8 fullkv_b projection and diagonalV copies replace U2048 V materialization. No new kernel, sharedSDK, Common or Native header layout.

This changes FP8 activation quantization/association and introduces aggregateBF16 rounding. CPU real-weight float32 algebra is not a vendor math emulator, device numerical qualification, performance result or Model quality proof. All GPU math remains BF16/F32, no GPUF64.

Original helper preserves globalrow128 scales and perhead448=K192+V256 interleave. Rotaries for the M8 helper come from the first8 selected positions, contribute only to discardedK and do not enter its V projection. This is different from sourceSTOP fullabsorbedMLA requiring576QK and a new512V attention primitive.

Complete query must include original freshprojections/append/metadataD2H/remap/IDsH2D/gather, unchanged QK/softmax, all aggregation/GEMM/FP8pack/unpack/scales/M8unusedK+allheadV/diagonalcopy, Oprojection/TP/copies/temporary destruction/drain. Load, reset and identical32768 prefixseeding are separately recorded, never erased from card cost.

First CPU: one production HOST TU replaces exact CPU802 consumer; all other object/dependency/provider bytes reused. A tiny pureCPU geometry executable consumes the same diagonal index function. No DSO dlopen/Runtime/weights/GPU in that CPU job. Device execution remains false until this actual receipt and a final bounded driver/claim are published. No fullModel or HTTP.
