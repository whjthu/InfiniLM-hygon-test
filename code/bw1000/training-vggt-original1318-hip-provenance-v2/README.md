# VGGT .1318 HIP provider evidence

The authenticated read of step `1103606.3188` found an 800-byte GPU log with SHA-256 `93d0621cd80157989a17e449a9089a782c0c1eada918bff5a285019c4c861160`. The first error was `ValueError: expected the actual named HIP shared object` in the original evidence function. The unchanged controller had reached that function after the original HIP device count and PCI calls. The log does not establish the library path or the cause of the name mismatch.

The overlay takes the original loaded HIP handle and identifies the executable mappings of the two actual assignment symbols. Both addresses must resolve to the same live named file mapping. It checks the mapped device and inode against the file it hashes, and it checks again after hashing. It supports the original AMD spelling and the Galaxy HIP spelling already present in the DTK tooling. Mere presence elsewhere in the memory map does not qualify a library.

`render.py` uses the exact frozen .1318 renderer and updates the staged controller, the provider evidence module, and their hashes in the old source index. The same scorer, inputs, resource limits and terminal auditor are retained. The original request is closed. The overlay produces a locally testable source bundle for a future independently approved request with its own placement and publication. No target provider observation has been made with this overlay.

Run `python -B -X utf8 check_local.py` in this source directory to exercise the original renderer and test the executable mapping rules.
