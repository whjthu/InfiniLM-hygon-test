# C4 history bank alias candidate

At a single-token, single-request step that creates no compressed key, the compressor gather already returns the complete BF16 bank consumed by attention. This default-OFF candidate holds that tensor through the original Work lifetime and uses it directly.

The original gather, FP8 conversion, indexer, attention and commit remain in the derived source. Boundary steps, multi-request inputs and histories larger than one gather chunk use the original allocation and copy. The four edits are reversible to the exact pinned CSA SHA in derive.py.

The policy must be set between completed serial requests. The Native setter rejects changes while any CSA forward is active. Counters report successful calls, eligible calls, alias calls and avoided copy bytes for each local device. Exceptions release the policy lease through RAII.

CPU_RESOURCE_CLAIM.json authorizes one 4-core, 8GiB, 300-second CPU build on f12r3n12 under parent 1087656. build_cpu.py checks the original 50 objects and library providers, verifies fresh -M against the original closure plus the new header, rebuilds one host unit, verifies actual MD equality and links 49 original objects. No Native load or device work is part of this claim.

The shared policy can be checked using g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -pthread check_policy.cpp -o policy-check, then running policy-check. Local results in SOURCE_PROOF.json cover all 32768 positions, invalid geometry and active-call/exception policy behavior.

Target numerical equivalence and performance are pending. A future GPU claim must pin the actual new DSO and CPU receipt, verify bank identity in the caller, retain the full original 4096 generation budget and context 32768, then compare matched OFF/ON timing. No quality or performance qualification is inferred from this CPU package.
