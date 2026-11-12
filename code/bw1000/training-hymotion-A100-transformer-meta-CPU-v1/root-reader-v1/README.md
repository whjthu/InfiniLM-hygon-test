# Original native HY-Motion Transformer meta CPU reader

Read-only stdlib reader of the single original request. Prefix CLAIM, REQUEST, EXPECTED_RESULT_SHA and SOURCES from the published source task and original transport receipt; execute on the original host using its Python3.12. This verifies exact original artifact/source identities and original PID/startticks absence. It never imports Torch, starts a job, loads weights/cache tensors or promotes training/model qualification.
