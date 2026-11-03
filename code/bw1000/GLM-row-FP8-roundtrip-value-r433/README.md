# GLM r433 private F32 value association

Default OFF, one private HOST consumer. Reuses the current r776 helper and original SDK; no new GPU kernel or shared ABI.

Original selected BF16 latent uses the same BlockQuant call. Identity FP8 linear with F32 output materializes the row roundtrip; probabilities are cast from the original BF16 values and both subsequent GEMMs remain FullFp32 until the final BF16 attention cast. Per-head F32 weights are newly owned, not borrowed from a nonexistent r809 decoded cache. First ON query includes all weight setup and fences.

The current e32 binary dtype2/mode1 float dispatch was inspected. Its complete historical build MD remains unknown; old cb28 device evidence is not inherited. This package has no device/Model qualification. Actual identity/quantization, precision, numerical outputs, complete wall and normal cleanup remain required.

CPU claim is four cores, 8GiB, 300 seconds: one new HOST compile/link, pure CPU identity/address contract and the new libffi/source glue. GPU and Model are false. Old 864 and prior suites are not rerun.
