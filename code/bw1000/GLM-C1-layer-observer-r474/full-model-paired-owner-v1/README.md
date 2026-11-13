# GLM-5.3 TP16 paired Native owner

`owner.py` is the owner from `full-model-screen32-v1` with the DSO binding replaced. It accepts the arm and a run verifier from the caller. The verifier must validate the new run manifest, the same frozen private plan and checkpoint metadata, and the selected arm. The controller, launch manifest, per-arm output directories and Slurm claim are separate work. This source carries no launch authorization.

The owner requires parent 1103606 and the I02 pair f13r1n05/f13r1n07. Each fresh process selects one of the two libraries in the independently verified link receipt from 1103606.59, checks its bytes, rejects a process that already maps either arm, and verifies the provider of the 12 typed C functions, four observer functions and profile getter before constructing the model. Deployment must preserve the 71-provider environment from the paired link and supply the original eight project DSO identities in CPU_BINDING.json for the runtime map check.

Both arms retain the original request loop, full checkpoint load, 32768 context, 4096 generation budget, 8 GiB floor, observer policy, scoring and normal teardown. Keep arm outputs and run identities disjoint. The 32 prompts may be paired for an engineering screen. One process lifetime per arm leaves process order and time confounded and cannot establish a formal performance win. The original Slurm 910371.906 result is independent of this package.

Run `python -B check_native_arm_CPU.py` locally from this source directory. The check uses the published terminal summary for provenance fields and synthetic files for hash and pre-load rejection. The actual ELF loading and dladdr checks require the target environment.
