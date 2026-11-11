# HOST complete FD / container kernel visibility CPU check

Original132 failed before explicitHIP SDK/model because container sameUIDFD
inspection encounteredPermissionError. This scope does not ignore that error
or claim a complete containerFD scan. HOST snapshots retain fail-closed full
kernel+sameUIDFD inspection. Container mode records only kernelKFDproc and
PCI/KFD topology, inheriting a fresh same-node/same-boot HOST fullFD check.
One metadata-only CPU step onT05 f11r4n09;2CPU4096MiB,0GPU,whole240s/body150s.
The CPU container deliberately attempts the old full scan for diagnosis and
recordsPermissionError distinctly, then validates the kernel-only observation.
No SDK/Torch import or device lease qualification; implicitcontainer helpers
remainUNKNOWN. Exact changedsnapshot bytes must be used in the next GPU scope,
which also requires a new fresh HOST check and actual8-device lease admission.
