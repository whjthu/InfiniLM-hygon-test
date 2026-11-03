# Kimi replicated low-rank M1 GEMV

Source-first default-OFF component; no existing Native/SDK/Common file is modified. Two independent actual layer1 BF16 checkpoint matrices, synthetic M1 inputs, one fixed block256/row candidate. Mode0 uses current `nn::Linear::forward`; mode1 changes only the dense dot implementation. This is not a full MoE, Model or quality result.

The first CPU claim compiles one thin HOST TU, one private GPU TU and a tiny pure shared-geometry contract. It checks immutable dependencies, private strong CABI, Bsymbolic/undefined relocations and exact MD. It never dlopens this library or creates a Runtime/device. Actual shader numerics, owner/device, canary, counter and end-to-end timing belong to a separately frozen single-card device entry.

All source/dependency evidence and explicit timing/stop boundaries are in PLAN.json and BUILD_CONTRACT.json. GPU_GO remains false in this CPU-stage package.
