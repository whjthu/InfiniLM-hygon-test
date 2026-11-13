"""Preparation only:call before bulk installation;does not launch or recover work."""
from pathlib import Path
from gencuda.frozen_bootstrap import bind_frozen_framework


def prepare_reference(workspace, *, request_id, claim_commit, source_manifest_SHA256,
                      frozen_root, manifest_bytes, manifest_SHA256, epoch, deadline):
    reference = bind_frozen_framework(
        Path(workspace), request_id=request_id, claim_commit=claim_commit,
        source_manifest_SHA256=source_manifest_SHA256,
        framework_root=Path(frozen_root), framework_manifest=manifest_bytes,
        framework_manifest_SHA256=manifest_SHA256, epoch=epoch, deadline=deadline,
    )
    # Future callers supply this actual read-only bind to the container. They must
    # still perform dependency/resource admission,one actual dispatch and terminal/post checks.
    return dict(reference=reference,
                container_bind=reference['framework_root'] + ':/run-scope/framework:ro',
                GPU_GO=False, Model_GO=False)
