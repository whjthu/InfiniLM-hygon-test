# Qwen3-VL public model relay

The four weights and ten metadata files match the original A100 files and the pinned official revision by full-file SHA-256. Transfer only these 14 paths over the existing finite relay. Both ends verify length and full-file hash while streaming. The receiver renames the staging directory only after all 14 files are verified. The full target identity still requires independent verification before use by the training entry.

The relay uses a single 12-hour deadline and a 1 MiB/s cap. The payload is 17545907231 bytes, so the cap alone implies at least 4.65 hours. GPU resources remain available during staging.
